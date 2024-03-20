from __future__ import annotations
from typing import Dict
from .parameter import Parameter
from jax.random import split

class ParameterStore:
    """The parameter store hosts a set of parameters, 
    letting you perform operations on a collection of parameters at once
    """
    def __init__(self, params : Dict[str, Parameter]) -> None:
        self.params = params

    def __len__(self):
        return len(self.params)

    def __getitem__(self, key):
        return self.params[key]

    def values(self):
        """Gets the values of the parameters in the store

        Returns:
            dict: the parameters
        """
        return {k: v.value for k,v in self.params.items()}


    def propose(self, key) -> ParameterStore:
        """Propose a new parameter store, containing newly sampled parameters

        Args:
            key (PRNGKey): a key to sample with

        Returns:
            ParameterStore: the newly sampled parameterstore
        """
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