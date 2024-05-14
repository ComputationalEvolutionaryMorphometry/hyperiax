from __future__ import annotations
from abc import ABC, abstractmethod

class Parameter(ABC):
    """ Base class for a parameter in a Markov Chain Monte Carlo simulation.
    """

    def __init__(self, value) -> None:
        """ Initialize the Parameter.

        :param value: The initial value of the parameter.
        """
        self.value = value

    @abstractmethod
    def update(self, value, accepted=True): 
        """ Update the value of the parameter, if accepted.

        value: The new value of the parameter.
        accepted: Whether the new value is accepted. Default true.
        """
        ...

    @abstractmethod
    def propose(self, key) -> Parameter: 
        """ Propose a new parameter given a key

        :param key: the key to generate the parameter with
        :return: A new parameter proposal
        """
        ...

    @abstractmethod
    def log_prior(self):
        """ Logarithm of the prior probability of the parameter.

        :return: The logarithm of the prior probability of the parameter.
        """
        ...

