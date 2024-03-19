from .parameter import Parameter

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

def inverse_gamma_logpdf(x, alpha, beta):
    """Log PDF of the inverse gamma distribution."""
    return alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1) * jnp.log(x) - beta / x

class VarianceParameter(Parameter):
    def __init__(self, value, alpha=3., beta=2.) -> None:
        super().__init__(value)
        self.alpha = alpha
        self.beta = beta

    def propose(self, key):
        # Use current variance as mode for proposal
        mode = self.value
        # Calculate alpha and beta for inverse gamma proposal distribution with the given mode
        alpha_proposal = 2.0  # Choose an arbitrary value for alpha
        beta_proposal = mode * (alpha_proposal + 1)

        return VarianceParameter(beta_proposal / jax.random.gamma(key, alpha_proposal, shape=()), self.alpha, self.beta)
    
    def update(self, value, accepted): 
        if accepted:
            self.value = value
    
    def log_prior(self):
        return inverse_gamma_logpdf(self.value,self.alpha,self.beta)
