from .parameter import Parameter

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.stats import uniform

def inverse_gamma_logpdf(x, alpha, beta):
    """ Log PDF of the inverse gamma distribution
    
    :param x: The value to evaluate the PDF at
    :param alpha: The alpha parameter of the inverse gamma distribution
    :param beta: The beta parameter of the inverse gamma distribution
    :return: The logarithm of the PDF of the inverse gamma distribution at x
    """
    return alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1) * jnp.log(x) - beta / x

def uniform_logpdf(x, a, b):
    """ Log PDF of the uniform distribution
    
    :param x: The value to evaluate the PDF at
    :param a: the left boundary of the uniform distribution
    :param b: the right boundary of the uniform distribution
    :return: The logarithm of the PDF of the uniform distribution at x
    """
    return uniform.logpdf(x, loc=a, scale=b-a)


class VarianceParameter(Parameter):
    def __init__(self, value, alpha=3., beta=2., a=0., b=1., proposal_var=.01, keep_constant=False, max=None, prior='inv_gamma', proposal='log_normal') -> None:
        """ Initialize a variance parameter
        
        :param value: The initial value of the parameter
        :param alpha: The alpha parameter of the inverse gamma distribution #sofia/morten change: and uniform and mirrored gaussian interval endpoints
        :param beta: The beta parameter of the inverse gamma distribution #sofia/morten change: and uniform and mirrored gaussian interval endpoints
        :param a: the left boundary of the uniform distribution and the mirrored gaussian 
        :param b: the right boundary of the uniform distribution and the mirrored gaussian
        :param proposal_var: The variance of the proposal distribution
        :param keep_constant: Whether the parameter should be kept constant in new proposals
        :param prior: 'inv_gamma' or 'uniform'
        :param proposal: 'log_normal' or 'mirrored_gaussian'
        """
        super().__init__(value)
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.proposal_var = proposal_var
        self.keep_constant = keep_constant
        self.max = max
        self.prior = prior
        self.proposal = proposal

    def propose(self, key): 
        '''
        Proposal distribution: 'lognormal' or 'mirorred_gaussian' 
        '''

        if self.keep_constant:
            return self
        
        if self.proposal == 'log_normal':
            shape = self.value.shape if hasattr(self.value,'shape') else ()
            new_value = jnp.exp(jnp.log(self.value) + jnp.sqrt(self.proposal_var)*jax.random.normal(key,shape=shape))
            if self.max is not None:
                new_value = jnp.clip(new_value, 0, self.max)
            return VarianceParameter(value=new_value, alpha=self.alpha, beta=self.beta,a=self.a, b=self.b, proposal_var=self.proposal_var, keep_constant=self.keep_constant, 
                                     max=self.max, prior=self.prior)
        
        elif self.proposal == 'mirrored_gaussian':
            x = self.value + jnp.sqrt(self.proposal_var)*jax.random.normal(key)
            while x<self.a or x>self.b:
                if x<self.a:
                    x = 2*self.a-x
                elif x>self.b:
                    x = 2*self.b-x
            return VarianceParameter(value=x, alpha=self.alpha, beta=self.beta, a= self.a, b=self.b, proposal_var=self.proposal_var, 
                                    keep_constant=self.keep_constant, max=self.max, prior=self.prior, proposal=self.proposal)
        else:
            raise ValueError(f"Unknown proposal type: {self.proposal}")


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
        if self.prior == 'inv_gamma':
            return inverse_gamma_logpdf(self.value, self.alpha, self.beta).sum()
        elif self.prior == 'uniform':
            return uniform_logpdf(self.value, self.a, self.b).sum()
        else:
            raise ValueError(f"Unknown prior type: {self.prior}")





