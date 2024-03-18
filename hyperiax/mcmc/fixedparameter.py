from .parameter import Parameter
from copy import copy
class FixedParameter(Parameter):
    def __init__(self, value) -> None:
        super().__init__(value)

    def propose(self, key):
        return copy(self)
    
    def update(self, value, accepted): 
        ...

    def log_prior(self): # p=1 since param is fixed
        return 0.