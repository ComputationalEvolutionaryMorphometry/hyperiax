from .parameter import Parameter
from copy import copy
class FixedParameter(Parameter):
    """A parameter that is constant regardless of updates.

    Will always have the value passed in the constructor
    """
    def __init__(self, value) -> None:
        super().__init__(value)

    def propose(self, key):
        return copy(self)
    
    def update(self, value, accepted): 
        ...

    def log_prior(self): # p=1 since param is fixed
        return 0.