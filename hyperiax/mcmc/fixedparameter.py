from .parameter import Parameter
from copy import copy
class FixedParameter(Parameter):
    """ A parameter that is constant regardless of updates.  The parameter always have the value passed in the constructor
    """
    def __init__(self, value) -> None:
        super().__init__(value)

    def propose(self, key):
        """ Propose a new value for the parameter

        :param key: A key to sample with
        :return: Copy of the parameter
        """
        return copy(self)
    
    def update(self, value, accepted): 
        """ Update the parameter. Does nothing as the parameter is fixed, i.e. the new value is discarded

        :param value: The new value to update the parameter with
        :param accepted: Whether the new value was accepted
        """
        ...

    def log_prior(self):
        """ Logarithm of the prior probability of the parameter

        :return: 0. as the parameter is fixed
        """
        return 0.

