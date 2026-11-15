"""Reproducible baseline NSGA-II and Pavel-reviewed 2-D ALSF variant.

Requires pymoo (installed on the user's machine / GitHub Actions).
Usage:
  python run_test1.py --mode baseline --generations 200 --seed 42
  python run_test1.py --mode alsf --generations 200 --seed 42
  python run_test1.py --mode alsf --generations 8 --pop-size 24 --flag-start 3 --tag smoke

Writes distinct run directories and REFUSES to overwrite existing archives.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_test1 import LOWER, UPPER, evaluate

ROOT = Path(__file__).resolve().parent
COLUMNS = ['X1', 'X2', 'Obj1', 'Obj2']


def main():
    p = argparse.ArgumentParser(description='ESCO 2026: Misi 2-D Test 1')
    p.add_argument('--mode', choices=['baseline', 'alsf'], required=True)
    p.add_argument('--generations', type=int, default=200)
    p.add_argument('--pop-size', type=int, default=100)
    p.add_argument('--offsprings', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tag', default='full')
    p.add_argument('--flag-start', type=int, default=50)
    p.add_argument('--tol', type=float, default=0.1,
                   help='Each coordinate tolerance; default matches original eps=0.1')
    p.add_argument('--objective-threshold', type=float, default=0.10)
    p.add_argument('--objective-floor', type=float, default=1e-3,
                   help='Explicit numerical policy for near-zero objective values')
    p.add_argument('--min-samples', type=int, default=10)
    args = p.parse_args()
    if args.offsprings is None:
        args.offsprings = args.pop_size
    if args.generations < 1 or args.pop_size < 4 or args.offsprings < 1:
        p.error('positive generations/offsprings and pop-size >= 4 required')

    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize
    from pymoo.core.callback import Callback

    outdir = ROOT/'results'/('test1_{}_seed{}_{}'.format(args.mode,args.seed,args.tag))
    archive_path = outdir/'archive.csv'
    process_path = outdir/'process.csv'
    pareto_path = outdir/'pareto.csv'
    if outdir.exists() and list(outdir.iterdir()):
        raise FileExistsError('Run outputs already exist: {}. Choose another --tag.'.format(outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'settings.json').write_text(json.dumps(vars(args),indent=2), encoding='utf8')

    class Test1Problem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2,n_obj=2,n_ieq_constr=0,n_eq_constr=0,
                             xl=LOWER,xu=UPPER,vtype=float)
            self.cache = {}
        def _evaluate(self, x, out, *a, **kw):
            coords = np.round(np.asarray(x, dtype=float), 8)
            key = tuple(coords.tolist())
            if key not in self.cache:
                f = evaluate(coords)
                self.cache[key] = tuple(float(v) for v in f)
                pd.DataFrame([[*coords,*f]],columns=COLUMNS).to_csv(
                    archive_path,mode='a',header=not archive_path.exists(),index=False)
            out['F'] = self.cache[key]

    class BaselineCallback(Callback):
        def notify(self, algorithm):
            pd.DataFrame([{'generation':algorithm.n_gen,
                           'evaluations':algorithm.evaluator.n_eval}]).to_csv(
                process_path, mode='a', header=not process_path.exists(), index=False)

    problem=Test1Problem()
    repair = None
    callback = BaselineCallback()
    if args.mode == 'alsf':
        from src.preprocessing import OffspringRepair
        from src.postprocessing import FlagCallback
        tolerances=np.array([args.tol,args.tol])
        repair=OffspringRepair(str(archive_path),problem.cache,1.0,
                               min_neighbours=1,decimal_places=8,
                               max_resample_attempts=1000,
                               rng=np.random.default_rng(args.seed),
                               tolerances=tolerances)
        callback=FlagCallback(str(archive_path),str(process_path),1.0,
                              args.objective_threshold,args.objective_threshold,
                              args.min_samples,0.0,args.flag_start,
                              repair,tolerances=tolerances,
                              objective_floors=(args.objective_floor,
                                                args.objective_floor))

    algorithm=NSGA2(pop_size=args.pop_size,n_offsprings=args.offsprings,
                     sampling=FloatRandomSampling(),
                     crossover=SBX(prob=0.9,eta=5),
                     mutation=PM(prob=1.0,eta=5),
                     repair=repair,eliminate_duplicates=True)
    #result=minimize(problem,algorithm,get_termination('n_gen',args.generations),
    #                seed=args.seed,callback=callback,save_history=False,verbose=True)
    result = minimize(
        problem,
        algorithm,
        get_termination('n_gen', args.generations),
        seed=args.seed,
        callback=callback,
        save_history=False,
        verbose=True,
        copy_algorithm=False
    )
    if result.X is None or result.F is None:
        raise RuntimeError('Optimization did not return a Pareto set')
    front=pd.DataFrame(np.column_stack((result.X,result.F)),columns=COLUMNS)
    if args.mode=='alsf' and archive_path.exists():
        a=pd.read_csv(archive_path)
        if 'flag' in a:
            lookup={tuple(np.round(x,8)):str(flag) for x,flag in zip(
                a[['X1','X2']].to_numpy(float),a['flag'])}
            front['flag']=[lookup.get(tuple(np.round(x,8)),'NEI') for x in result.X]
    front.to_csv(pareto_path,index=False)
    report={'mode':args.mode,'seed':args.seed,'generations':args.generations,
            'pymoo_evaluations':int(result.algorithm.evaluator.n_eval),
            'unique_archive_points':len(problem.cache),
            'pareto_size':len(front)}
    if args.mode == 'alsf':
        if not process_path.exists():
            raise RuntimeError('ALSF process.csv was not produced; verify callback')
        history = pd.read_csv(process_path)
        expected = ['n_repair_calls','n_offspring_seen',
                    'n_forbidden_candidates','n_regenerated','n_attempts']
        missing = [col for col in expected if col not in history]
        if missing:
            raise RuntimeError(f'ALSF diagnostic columns missing: {missing}')
        report.update({f'total_{c}':int(history[c].sum()) for c in expected})
        report['final_exclusion_centers'] = int(history['n_exclusion_centers'].iloc[-1])
        if report['total_n_offspring_seen'] == 0:
            raise RuntimeError('ALSF repair never saw an offspring: invalid integration')
        if report['total_n_forbidden_candidates'] == 0:
            print('WARNING: no forbidden candidates were seen; '
                  'repair had no demonstrated effect on this run.')
    (outdir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('\nRESULTS:',outdir,'\n',json.dumps(report,indent=2))

if __name__=='__main__':
    main()
