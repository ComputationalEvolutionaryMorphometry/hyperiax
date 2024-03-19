from __future__ import annotations
from abc import ABC, abstractmethod

class Parameter(ABC):
    def __init__(self, value) -> None:
        self.value = value

    @abstractmethod
    def update(self, value, accepted): 
        """update the value of the parameter using the accept into
        """
        ...

    @abstractmethod
    def propose(self, key) -> Parameter: 
        """Propose a new parameter given a key

        Args:
            key (PRNGKey): the key to generate the parameter with

        Returns:
            Parameter: the new parameter generated
        """
        ...

    @abstractmethod
    def log_prior(self):
        pass