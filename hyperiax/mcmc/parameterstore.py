from __future__ import annotations
from typing import Dict
from .parameter import Parameter
from jax.random import split

from copy import deepcopy

class ParameterStore:
    def __init__(self, params : Dict[str, Parameter]) -> None:
        self.params = params

    def __len__(self):
        return len(self.params)

    def __getitem__(self, key):
        return self.params[key]

    def values(self):
        return {k: v.value for k,v in self.params.items()}

    def propose(self, key):
        pcount = len(self)
        subkeys = split(key, pcount)

        return {k: v.propose(rngkey) for rngkey, (k,v) in zip(subkeys, self.params.items())}

    def propose(self, key) -> ParameterStore:
        pcount = len(self)
        subkeys = split(key, pcount)
        
        return ParameterStore({
            k: v.propose(rngkey)
            for 
            rngkey, (k,v)
            in 
            zip(subkeys, self.params.items())
        })

    def log_prior(self):
        return sum([v.log_prior() for v in self.params.values()])