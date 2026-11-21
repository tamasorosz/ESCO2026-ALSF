"""Pre-evaluation ALSF rejection, without replacement or objective calls.

pymoo 0.6.x lifecycle: initialization remains unfiltered. A single ordinary
mating.do call produces the duplicate-eliminated proposal batch. Filtering is
applied AFTER that call; rejected offspring never re-enter mating/evaluation.
An empty accepted Population advances the generation with unchanged parents.
"""
from __future__ import annotations
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from src.find_robust_clusters import decision_columns
from src.offspring_policy import VerifiedViolationFilter


class OffspringDropFilter:
    def __init__(self,tolerances,radius=1.0):
        self.policy=VerifiedViolationFilter(tolerances,radius=radius)
        self.last_counts=dict(n_proposed=0,n_dropped=0,n_accepted=0)

    def update_flags(self,archive):
        if 'verified_violation' not in archive:
            raise ValueError('Expected verified_violation evidence')
        self.policy.update(archive[decision_columns(archive)].to_numpy(float),
                           archive.verified_violation.to_numpy(bool))

    def keep_mask(self,coords):
        coords=np.asarray(coords,dtype=float)
        if coords.ndim!=2 or not np.isfinite(coords).all():
            raise ValueError('Expected finite coordinates [n,d]')
        keep=np.array([not self.policy.forbidden(x) for x in coords],dtype=bool)
        self.last_counts=dict(n_proposed=len(coords),n_dropped=int((~keep).sum()),
                              n_accepted=int(keep.sum()))
        return keep

    def filter_population(self,population):
        if not len(population):
            self.last_counts=dict(n_proposed=0,n_dropped=0,n_accepted=0)
            return Population.empty()
        return population[self.keep_mask(population.get('X'))]


class DropNSGA2(NSGA2):
    """Variable-size offspring infill; generation budget handles all-drop cases."""
    def __init__(self,drop_filter,**kwargs):
        super().__init__(**kwargs)
        self.drop_filter=drop_filter
        self.last_drop_counts=dict(n_proposed=0,n_dropped=0,n_accepted=0)
        self.mating_batches=0

    def _infill(self):
        # Do not call GeneticAlgorithm._infill: it force-terminates on empty
        # batches. Mating may retry ordinary duplicates, never ALSF rejections.
        proposed=self.mating.do(self.problem,self.pop,self.n_offsprings,
                    algorithm=self,random_state=self.random_state)
        self.mating_batches+=1
        accepted=self.drop_filter.filter_population(proposed)
        self.last_drop_counts=self.drop_filter.last_counts.copy()
        return accepted

    def _advance(self,infills=None,**kwargs):
        if infills is None or len(infills)==0:
            # Return None so Algorithm.advance still calls _post_advance:
            # termination, callback and generation counters all progress.
            return None
        return super()._advance(infills=infills,**kwargs)
