from __future__ import annotations
from typing import Dict
from .parameter import Parameter
from jax.random import split

class ParameterStore:
    """ The parameter store hosts a set of parameters. 
    """

    def __init__(self, params : Dict[str, Parameter]) -> None:
        """ Initialize the parameter store.

        :param params: Dict with the parameters to store and their associated keys (names).
        """
        self.params = params

    def __len__(self):
        """ Get the number of parameters in the store.

        :return: The number of parameters in the store
        """
        return len(self.params)

    def __getitem__(self, key):
        return self.params[key]

    def values(self):
        """ Gets the values of the parameters in the store

        :return: The parameters in a dictionary.
        """
        return {k: v.value for k,v in self.params.items()}


    def propose(self, key) -> ParameterStore:
        """ Propose a new parameter store, containing newly sampled parameters.

        :param key: A key to sample with.
        :return: The newly sampled parameter store.
        """
        pcount = len(self)
        subkeys = split(key, pcount)
        
        proposals_and_corrections = {k: v.propose(rngkey) for rngkey,(k,v) in zip(subkeys,self.params.items())}
        new_params = {k: v[0] for k,v in proposals_and_corrections.items()}
        log_correction = sum(v[1] for v in proposals_and_corrections.values())
        return ParameterStore(new_params),log_correction

    def log_prior(self):
        """ Logarithm of the prior probability of the parameters.

        :return: The logarithm of the prior probability of the parameters.
        """
        return sum([v.log_prior() for v in self.params.values()])

