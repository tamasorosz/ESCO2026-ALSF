"""Read-only peak screening instrumentation; inherited policies remain unchanged.

Radius .18 in decision units is fixed before new runs. Sparse means fewer than
10 archived neighbours in the scaled Euclidean tolerance ball, NOT a certified
NEI classification of an unevaluated proposal. Replacement checks are included.
"""
import numpy as np
from scipy.spatial import cKDTree
from src.preprocessing import OffspringRepair
from src.offspring_drop import OffspringDropFilter

class Observer:
    def record(self,x,rejected,policy):
        if np.linalg.norm(np.asarray(x)-2.5)>.18:return
        coords=self.observation['coords']
        if getattr(self,'_coords_id',None)!=id(coords):
            self._coords_id=id(coords);self._tree=cKDTree(coords/.1) if len(coords) else None
        count=0 if self._tree is None else len(self._tree.query_ball_point(np.asarray(x)/.1,1.))
        self.events.append(dict(generation=self.observation['generation'],policy=policy,
            X1=float(x[0]),X2=float(x[1]),rejected=bool(rejected),archive_neighbors=count,
            allowed_sparse=bool(not rejected and count<10)))

class ObservedRepair(Observer,OffspringRepair):
    def __init__(self,observation,events,*args,**kwargs):
        self.observation=observation;self.events=events
        super().__init__(*args,**kwargs)
    def _forbidden(self,x):
        rejected=super()._forbidden(x);self.record(x,rejected,'repair_check');return rejected

class ObservedDrop(Observer,OffspringDropFilter):
    def __init__(self,observation,events,*args,**kwargs):
        self.observation=observation;self.events=events
        super().__init__(*args,**kwargs)
    def keep_mask(self,coords):
        keep=super().keep_mask(coords)
        for x,k in zip(coords,keep):self.record(x,not k,'drop_proposal')
        return keep
