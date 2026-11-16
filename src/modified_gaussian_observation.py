"""Passive instrumentation of unchanged inherited rejection policies."""
import numpy as np
from src.preprocessing import OffspringRepair
from src.offspring_drop import OffspringDropFilter

class ObservedRepair(OffspringRepair):
    def __init__(self,observation,events,*args,**kwargs):
        self.observation=observation;self.events=events;self.replacement=False
        super().__init__(*args,**kwargs)
    def _do(self,problem,X,**kwargs):
        self.replacement=False
        return super()._do(problem,X,**kwargs)
    def _forbidden(self,x):
        rejected=super()._forbidden(x)
        if self.observation['mode']=='resample':
            self.events.append((self.observation['generation'],'replacement' if self.replacement else 'original',
                float(x[0]),float(x[1]),bool(rejected)))
        self.replacement=bool(rejected)
        return rejected

class ObservedDrop(OffspringDropFilter):
    def __init__(self,observation,events,*args,**kwargs):
        self.observation=observation;self.events=events
        super().__init__(*args,**kwargs)
    def keep_mask(self,coords):
        keep=super().keep_mask(coords)
        for x,k in zip(coords,keep):
            self.events.append((self.observation['generation'],'original',float(x[0]),float(x[1]),bool(not k)))
        return keep
