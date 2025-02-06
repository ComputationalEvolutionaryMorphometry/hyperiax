from .parameter import Parameter

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.stats import uniform
import matplotlib.pyplot as plt

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
    def __init__(self, value, alpha=3., beta=2., min=None, max=None, proposal_var=.01, keep_constant=False, prior='inv_gamma', proposal='log_normal') -> None:
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
        self.min = min
        self.max = max
        self.proposal_var = proposal_var
        self.keep_constant = keep_constant
        self.prior = prior
        self.proposal = proposal

    def propose(self, key): 
        '''
        Proposal distribution: 'lognormal' or 'mirorred_gaussian' 
        '''

        if self.keep_constant:
            return self,0.
        
        if self.proposal == 'log_normal':
            shape = self.value.shape if hasattr(self.value,'shape') else ()
            noise = jnp.sqrt(self.proposal_var)*jax.random.normal(key,shape=shape)
            new_value = jnp.exp(jnp.log(self.value)+noise)
            if self.max is not None:
                new_value = jnp.clip(new_value,0,self.max)
            # Add correction term for log-normal proposal asymmetry
            log_correction = -noise  # Since q(x'|x)/q(x|x') = exp(-eps) where eps is the noise
            return VarianceParameter(**{**self.__dict__,'value':new_value}),log_correction
        
        elif self.proposal == 'mirrored_gaussian':
            shape = self.value.shape if hasattr(self.value,'shape') else ()
            new_value = self.value + jnp.sqrt(self.proposal_var)*jax.random.normal(key,shape=shape)
            # Mirror values outside bounds component-wise
            if self.min is not None:
                below_min = new_value<self.min
                new_value = jnp.where(below_min,2*self.min-new_value,new_value)
            if self.max is not None:
                above_max = new_value>self.max
                new_value = jnp.where(above_max,2*self.max-new_value,new_value)
            log_correction = 0.
            return VarianceParameter(**{**self.__dict__,'value':new_value}),log_correction
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
            return uniform_logpdf(self.value, self.min, self.max).sum()
        else:
            raise ValueError(f"Unknown prior type: {self.prior}")

    def plot_prior(self, x_min=1e-3, x_max=1, n_points=100):
            """Plot the prior distribution of the parameter
            
            :param x_min: Minimum x value to plot
            :param n_points: Number of points to plot
            :param x_max: Maximum x value to plot
            """
            # Create array of parameter values
            x = jnp.linspace(x_min,x_max,n_points)

            # Calculate prior for each value
            prior_vals = [VarianceParameter(**{**self.__dict__,'value':val}).log_prior() for val in x]

            # Plot exp(log_prior)
            plt.figure()
            plt.plot(x,jnp.exp(jnp.array(prior_vals)))
            plt.xlabel('Parameter value')
            plt.ylabel('Prior density')
            plt.title('Prior distribution')

            # Find x value where prior is maximized
            max_idx = jnp.argmax(jnp.exp(jnp.array(prior_vals)))
            max_x = x[max_idx]
            print(f"Prior is maximized at x = {max_x}")
