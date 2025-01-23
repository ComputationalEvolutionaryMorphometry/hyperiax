from .parameter import Parameter

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

def inverse_gamma_logpdf(x, alpha, beta):
    """ Log PDF of the inverse gamma distribution
    
    :param x: The value to evaluate the PDF at
    :param alpha: The alpha parameter of the inverse gamma distribution
    :param beta: The beta parameter of the inverse gamma distribution
    :return: The logarithm of the PDF of the inverse gamma distribution at x
    """
    return alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1) * jnp.log(x) - beta / x

class VarianceParameter(Parameter):
    def __init__(self, value, alpha=3., beta=2., proposal_var=.01, keep_constant=False, max=None) -> None:
        """ Initialize a variance parameter
        
        :param value: The initial value of the parameter
        :param alpha: The alpha parameter of the inverse gamma distribution
        :param beta: The beta parameter of the inverse gamma distribution
        :param proposal_var: The variance of the proposal distribution
        :param keep_constant: Whether the parameter should be kept constant in new proposals
        """
        super().__init__(value)
        self.alpha = alpha
        self.beta = beta
        self.proposal_var = proposal_var
        self.keep_constant = keep_constant
        self.max = max

    def propose(self, key):
        """ Propose a new value for the parameter
        
        :param key: A key to sample with
        :return: The newly proposed parameter
        """
        if self.keep_constant:
            return self

        shape = self.value.shape if hasattr(self.value,'shape') else ()
        new_value = jnp.exp(jnp.log(self.value)+jnp.sqrt(self.proposal_var)*jax.random.normal(key,shape=shape))
        if self.max is not None:
            new_value = jnp.clip(new_value, 0, self.max)
        return VarianceParameter(new_value, self.alpha, self.beta, self.proposal_var, self.keep_constant, self.max)
    
    def update(self, value, accepted): 
        """ Update the parameter with a new value
        
        :param value: The new value to update the parameter with
        :param accepted: Whether the new value was accepted.
        """
        if accepted:
            if self.max is not None:
                self.value = jnp.clip(value, 0, self.max)
            else:
                self.value = value
    
    def log_prior(self):
        """ Logarithm of the prior probability of the parameter
        
        :return: The logarithm of the prior probability of the parameter
        """
        return inverse_gamma_logpdf(self.value,self.alpha,self.beta).sum()
