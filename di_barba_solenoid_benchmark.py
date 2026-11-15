"""Analytical 10-D multiobjective solenoid benchmark inspired by Di Barba (2016).

Reference:
    P. Di Barba, "Multi-objective wind-driven optimisation and magnet design",
    Electronics Letters 52 (2016), 1216-1218, doi:10.1049/el.2016.1527.

Verified article data: a solenoid of half-length 15 mm, radii 9/10 mm,
control interval +-5 mm, 20 axial sections, 10 independent symmetric
current-density variables, 41 axial observation positions, target B = 0.84 mT.

ASSUMPTIONS / NOT A BIT-EXACT REPRODUCTION:
    * Currents are independent, signed and uniformly distributed in each
      rectangular winding cross section.
    * Adjustable bounds are +-2 MA/m2; the article's bounds are not confirmed.
    * Objectives are normalized mean squared field error and normalized J^2.
      They represent the published field residual / power-loss tradeoff,
      but the source equation images could not be checked for exact factors.
    * Resistivity is used only to report indicative ohmic power in Watts.

The field of each finite annular section is integrated in closed form along
its axis, treating the azimuthal volume current density as uniform.

Design coordinates x_i = J_i / J_SCALE, so x_i are dimensionless and
suitable for ten-dimensional distance calculations in ALSF.

No FEM or pymoo dependency is required to use this model and construct an
independent bounded, convex weighted-sum reference Pareto front.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class SolenoidSettings:
    half_length: float = 15e-3
    inner_radius: float = 9e-3
    outer_radius: float = 10e-3
    control_half_length: float = 5e-3
    n_sections: int = 20
    n_field_samples: int = 41
    target_field: float = 0.84e-3
    j_scale: float = 1e6                # A/m^2: normalized decision variables
    j_min: float = -2e6                # A/m^2: illustrative, not source-verified
    j_max: float = 2e6                 # A/m^2: illustrative, not source-verified
    copper_resistivity: float = 1.724e-8  # ohm*m at ~20 Celsius, illustrative
    independent_j_tolerance: float = 0.1e6  # A/m^2, illustrative


class DiBarbaSolenoid:
    def __init__(self, settings: SolenoidSettings | None = None):
        s = settings or SolenoidSettings()
        if s.n_sections % 2 != 0:
            raise ValueError("Number of sections must be even.")
        if not (0 < s.inner_radius < s.outer_radius):
            raise ValueError("Expected 0 < inner_radius < outer_radius.")
        if not (0 < s.control_half_length <= s.half_length):
            raise ValueError("Control interval must lie within the solenoid.")
        if not (s.j_min < 0 < s.j_max and s.j_scale > 0):
            raise ValueError("Bounds must contain zero; J_SCALE must be positive.")
        self.s = s
        self.n_var = s.n_sections // 2
        self.z_samples = np.linspace(-s.control_half_length,
                                     s.control_half_length, s.n_field_samples)
        self.edges = np.linspace(-s.half_length, s.half_length, s.n_sections + 1)
        self.lower = np.full(self.n_var, s.j_min / s.j_scale)
        self.upper = np.full(self.n_var, s.j_max / s.j_scale)

        # A[m, i] = B at observation m from J_i = 1 A/m^2, both symmetric sections.
        self.A = np.zeros((s.n_field_samples, self.n_var))
        for i in range(self.n_var):
            left = self._section_field_kernel(self.edges[i], self.edges[i + 1])
            j = s.n_sections - 1 - i
            right = self._section_field_kernel(self.edges[j], self.edges[j + 1])
            self.A[:, i] = left + right

        # K maps dimensionless currents to B/B0.
        self.K = self.A * (s.j_scale / s.target_field)
        self.b_target = np.ones(s.n_field_samples)
        self.winding_volume = (np.pi * (s.outer_radius**2 - s.inner_radius**2)
                               * 2 * s.half_length)
        self.p_ref = s.copper_resistivity * self.winding_volume * s.j_scale**2

    @staticmethod
    def _antiderivative(r: float, u: np.ndarray) -> np.ndarray:
        """F(r,u) = u log[r + sqrt(r^2+u^2)], r>0.

        Its difference over two radii equals the radial integral of
        u/sqrt(r^2+u^2). At u=0 this formula is finite and exactly zero.
        """
        return u * np.log(r + np.hypot(r, u))

    def _section_field_kernel(self, z0: float, z1: float,
                              y: np.ndarray | None = None) -> np.ndarray:
        """Axial B(y) in T for a finite annular section, J=1 A/m^2."""
        s = self.s
        if y is None:
            y = self.z_samples
        u1, u0 = z1 - y, z0 - y
        F = self._antiderivative
        return (0.5 * MU0 *
                (F(s.outer_radius, u1) - F(s.inner_radius, u1)
                 - F(s.outer_radius, u0) + F(s.inner_radius, u0)))

    def field(self, x: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        """Axial field in T. x = 10 dimensionless independent symmetric J's."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_var:
            raise ValueError(f"Expected {self.n_var} decision variables.")
        if z is None:
            return self.A @ (self.s.j_scale * x)
        z = np.asarray(z, dtype=float).reshape(-1)
        result = np.zeros(z.shape, dtype=float)
        for i in range(self.n_var):
            j = self.s.n_sections - 1 - i
            result += (self._section_field_kernel(self.edges[i], self.edges[i + 1], z)
                       + self._section_field_kernel(self.edges[j], self.edges[j + 1], z)) * self.s.j_scale * x[i]
        return result

    def evaluate(self, x: np.ndarray) -> tuple[float, float]:
        """Minimize normalized field MSE and normalized ohmic power proxy."""
        x = np.asarray(x, dtype=float).reshape(self.n_var)
        residual = self.K @ x - self.b_target
        f1 = np.mean(residual**2)
        f2 = np.mean(x**2)
        return float(f1), float(f2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """2x10 analytical Jacobian of (f1,f2) wrt dimensionless x."""
        x = np.asarray(x, dtype=float).reshape(self.n_var)
        g1 = (2 / self.s.n_field_samples) * self.K.T @ (self.K @ x - self.b_target)
        g2 = (2 / self.n_var) * x
        return np.vstack([g1, g2])

    def ohmic_loss_watt(self, x: np.ndarray) -> float:
        """Illustrative copper loss; all sections assumed fully copper-filled."""
        return self.evaluate(x)[1] * self.p_ref

    def current_density(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) * self.s.j_scale

    def full_winding_j(self, x: np.ndarray) -> np.ndarray:
        """20 signed current densities, symmetric around z=0."""
        j = self.current_density(x)
        return np.concatenate((j, j[::-1]))

    def hessians(self):
        """Constant positive-semidefinite Hessians of the defined objectives."""
        return np.stack([2*self.K.T@self.K/self.s.n_field_samples,
                         2*np.eye(self.n_var)/self.n_var])

    def _weighted_solution(self, lam):
        n = self.s.n_field_samples
        A = np.vstack([self.K/np.sqrt(n), np.sqrt(lam/self.n_var)*np.eye(self.n_var)])
        b = np.r_[self.b_target/np.sqrt(n), np.zeros(self.n_var)]
        res = lsq_linear(A, b, bounds=(self.lower,self.upper), tol=1e-12, max_iter=1000)
        if not res.success or not np.isfinite(res.x).all():
            raise RuntimeError(f'Reference solve failed: lambda={lam}: {res.message}')
        x = res.x
        g = self.gradient(x)[0]+lam*self.gradient(x)[1]
        # Projected gradient mapping is zero exactly at box-constrained KKT points.
        kkt = np.max(abs(x-np.clip(x-g,self.lower,self.upper)))
        return dict(zip([f'X{i+1}' for i in range(self.n_var)],x),
                    Obj1=self.evaluate(x)[0],Obj2=self.evaluate(x)[1],
                    **{'lambda':float(lam)},solver_success=bool(res.success),
                    solver_status=int(res.status),solver_iterations=int(res.nit),
                    solver_optimality=float(res.optimality),kkt_residual=float(kkt),
                    bound_violation=float(max(np.max(self.lower-x),np.max(x-self.upper),0)))

    def reference_pareto(self, n_weights=241, *, adaptive_gap=None, max_points=1200):
        """Convex bounded weighted least squares, with optional objective-gap refinement.

        All scalarizations are globally solved to numerical tolerance. Deduplication
        and strict nondominance filtering apply only to the returned plotting front;
        every solve is retained in self.reference_diagnostics. Gap distances use
        objective ranges of the initial solved front, fixed throughout refinement.
        """
        if n_weights < 2 or (adaptive_gap is not None and adaptive_gap <= 0):
            raise ValueError('Need at least two weights and a positive adaptive gap')
        records = [self._weighted_solution(lam) for lam in np.r_[0.,np.logspace(-8,7,n_weights)]]
        zero = dict(zip([f'X{i+1}' for i in range(self.n_var)], np.zeros(self.n_var)),
                    Obj1=1., Obj2=0., **{'lambda':np.inf}, solver_success=True,
                    solver_status=3,solver_iterations=0,solver_optimality=0.,
                    kkt_residual=0.,bound_violation=0.)
        records.append(zero)
        scale = np.maximum(np.ptp(np.array([[r['Obj1'],r['Obj2']] for r in records]),axis=0),1e-12)
        def clean(rows):
            df = pd.DataFrame(rows).sort_values(['Obj1','Obj2']).reset_index(drop=True)
            keep=[]; best=np.inf
            for i,r in df.iterrows():
                if r.Obj2 < best:
                    keep.append(i); best=r.Obj2
            df=df.iloc[keep].reset_index(drop=True)
            # Preserve exact zero-loss endpoint; near duplicates are display-only.
            keep=[len(df)-1]
            for i in range(len(df)-2,-1,-1):
                if np.max(abs((df.iloc[i][['Obj1','Obj2']].to_numpy(float)-
                              df.iloc[keep[-1]][['Obj1','Obj2']].to_numpy(float))/scale))>1e-10:
                    keep.append(i)
            return df.iloc[keep[::-1]].reset_index(drop=True)
        front=clean(records)
        if adaptive_gap is not None:
            for _ in range(16):
                xy=front[['Obj1','Obj2']].to_numpy()/scale
                gap=np.linalg.norm(np.diff(xy,axis=0),axis=1)
                selected=np.flatnonzero(gap>adaptive_gap)
                if not len(selected) or len(records)>=max_points:
                    break
                selected=selected[np.argsort(gap[selected])[::-1]][:max_points-len(records)]
                existing={r['lambda'] for r in records}
                added=[]
                for i in selected:
                    left,right=sorted(front.iloc[[i,i+1]]['lambda'].to_numpy())
                    lam=(right/10 if left==0 else left*10 if np.isinf(right) else np.sqrt(left*right))
                    if lam not in existing:
                        added.append(self._weighted_solution(lam));existing.add(lam)
                if not added:
                    break
                records.extend(added);front=clean(records)
        self.reference_diagnostics=pd.DataFrame(records)
        return front

    def box_objective_extrema(self, x, delta_j=None):
        """Exact convex-box extrema, up to least-squares numerical tolerance.

        f1 maximum is at a vertex, f1 minimum is bounded least squares;
        f2 extrema are separable. No padding outside the feasible box is used.
        """
        x=np.asarray(x,dtype=float).reshape(self.n_var)
        step=(self.s.independent_j_tolerance if delta_j is None else delta_j)/self.s.j_scale
        if not np.isfinite(x).all() or np.any(x<self.lower) or np.any(x>self.upper):
            raise ValueError('Sensitivity center must be feasible and finite')
        if not np.isfinite(step) or step<0:
            raise ValueError('delta_j must be finite and nonnegative')
        if step==0:
            values=np.asarray(self.evaluate(x))
            return values,values.copy()
        lo=np.maximum(self.lower,x-step);hi=np.minimum(self.upper,x+step)
        n=1<<self.n_var
        bits=((np.arange(n,dtype=np.uint32)[:,None]>>np.arange(self.n_var,dtype=np.uint32))&1)
        corners=lo+bits*(hi-lo)
        residuals=corners@self.K.T-self.b_target
        f1_max=np.mean(residuals**2,axis=1).max()
        result=lsq_linear(self.K,self.b_target,bounds=(lo,hi),tol=1e-12,max_iter=1000)
        if not result.success:
            raise RuntimeError('Sensitivity minimum solve failed: '+result.message)
        f1_min=self.evaluate(result.x)[0]
        near=np.where((lo<=0)&(hi>=0),0.,np.minimum(abs(lo),abs(hi)))
        far=np.maximum(abs(lo),abs(hi))
        return np.array([f1_min,np.mean(near**2)]),np.array([f1_max,np.mean(far**2)])

    def exact_box_sensitivity(self, x, delta_j=None):
        """Legacy absolute objective deviations, preserved for existing plots."""
        low,high=self.box_objective_extrema(x,delta_j)
        value=np.asarray(self.evaluate(x))
        return tuple(np.maximum(abs(low-value),abs(high-value)))

    def exact_relative_box_sensitivity(self, x, delta_j=None, floors=(1e-3,1e-3)):
        """Maximal symmetric relative variations used by ALSF, not absolute deltas."""
        from src.local_variation import relative_pairwise_variation
        low,high=self.box_objective_extrema(x,delta_j)
        value=np.asarray(self.evaluate(x))
        return np.maximum(relative_pairwise_variation(value,low,floors),
                          relative_pairwise_variation(value,high,floors))


if __name__ == "__main__":
    model = DiBarbaSolenoid()
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    ref = model.reference_pareto(n_weights=101)
    path = out / "pareto_di_barba_analytical_reference.csv"
    ref.to_csv(path, index=False)
    print("Analytical solenoid model | independent normalized 10-D currents")
    print("A shape:", model.A.shape)
    print("Target field:", model.s.target_field * 1e3, "mT")
    print("Current density bounds:", model.s.j_min, model.s.j_max, "A/m^2")
    print("Power reference:", model.p_ref, "W")
    print("Zero-current objectives:", model.evaluate(np.zeros(model.n_var)))
    print("Reference solutions:", len(ref))
    print("Reference CSV:", path)
