from .parameter import Parameter

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

def inverse_gamma_logpdf(x, alpha, beta):
    """Log PDF of the inverse gamma distribution."""
    return alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1) * jnp.log(x) - beta / x

class VarianceParameter(Parameter):
    def __init__(self, value, alpha=3., beta=2., proposal_var=.01, keep_constant=False) -> None:
        super().__init__(value)
        self.alpha = alpha
        self.beta = beta
        self.proposal_var = proposal_var
        self.keep_constant = keep_constant

    def propose(self, key):
        if self.keep_constant:
            return self
        ## Use current variance as mode for proposal
        #mode = self.value
        ## Calculate alpha and beta for inverse gamma proposal distribution with the given mode
        #alpha_proposal = 2.0  # Choose an arbitrary value for alpha
        #beta_proposal = mode * (alpha_proposal + 1)
        #return VarianceParameter(beta_proposal / jax.random.gamma(key, alpha_proposal, shape=()), self.alpha, self.beta)

        return VarianceParameter(jnp.exp(jnp.log(self.value)+jnp.sqrt(self.proposal_var)*jax.random.normal(key)), self.alpha, self.beta, self.proposal_var, self.keep_constant)
    
    def update(self, value, accepted): 
        if accepted:
            self.value = value
    
    def log_prior(self):
        return inverse_gamma_logpdf(self.value,self.alpha,self.beta)
