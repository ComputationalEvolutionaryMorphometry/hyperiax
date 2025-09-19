from __future__ import annotations
from typing import Dict, Tuple

from jax import Array
from jax.typing import ArrayLike
from jax.random import split

from .parameters import Parameter


class ParameterStore:
    """
    The parameter container that hosts a set of parameters.
    """

    def __init__(self, params_dict: Dict[str, Parameter]) -> None:
        """
        Initialize the parameter store.

        Args:
            params_dict (Dict[str, Parameter]): A dictionary of parameters to store.
        """
        self.params_dict = params_dict

    def __len__(self) -> int:
        """
        Gets the number of parameters in the store.

        Returns:
            int: The number of parameters in the store.
        """
        return len(self.params_dict)

    def __getitem__(self, key: str) -> Parameter:
        """
        Gets a parameter from the store by key.

        Args:
            key (str): The key of the parameter to retrieve.

        Returns:
            Parameter: The parameter associated with the given key.
        """
        return self.params_dict[key]

    def values(self) -> Dict[str, ArrayLike]:
        """
        Get the current values of all parameters in the store.

        Returns:
            Dict[str, ArrayLike]: A dictionary of parameter values.
        """
        return {k: v.value for k, v in self.params_dict.items()}

    def propose(self, rng_key: Array) -> Tuple[ParameterStore, ArrayLike]:
        """
        Propose a new parameter store, containing newly sampled parameters.

        Args:
            rng_key (Array): A key to sample with.

        Returns:
            ParameterStore: The newly sampled parameter store.
        """
        pcount = len(self)
        sub_keys = split(rng_key, pcount)

        proposals_and_corrections = {
            k: v.propose(key) for key, (k, v) in zip(sub_keys, self.params_dict.items())
        }
        new_params_dict = {k: v[0] for k, v in proposals_and_corrections.items()}
        log_correction = sum(v[1] for v in proposals_and_corrections.values())
        return ParameterStore(new_params_dict), log_correction

    def log_prior(self) -> ArrayLike:
        """
        Logarithm of the joint prior probability of the whole set of parameters.

        Returns:
            ArrayLike: The logarithm of the joint prior probability of the whole set of parameters.
        """
        return sum([v.log_prior() for v in self.params_dict.values()])
