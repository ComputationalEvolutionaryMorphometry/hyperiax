from .parameter import Parameter

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.stats import uniform
import matplotlib.pyplot as plt

def uniform_logpdf(x, a, b):
    """ Log PDF of the uniform distribution
    
    :param x: The value to evaluate the PDF at
    :param a: the left boundary of the uniform distribution
    :param b: the right boundary of the uniform distribution
    :return: The logarithm of the PDF of the uniform distribution at x
    """
    return uniform.logpdf(x, loc=a, scale=b-a)


class FlatParameter(Parameter):
    def __init__(self, value, min=None, max=None, proposal_var=.01, keep_constant=False) -> None:
        """ Initialize a parameter with flat prior
        
        :param value: The initial value of the parameter
        :param min: the left boundary of the interval
        :param max: the right boundary of the interval
        :param proposal_var: The variance of the proposal distribution
        :param keep_constant: Whether the parameter should be kept constant in new proposals
        """
        super().__init__(value)
        self.min = min
        self.max = max
        self.proposal_var = proposal_var
        self.keep_constant = keep_constant

    def propose(self, key): 
        '''
        Proposal distribution: 'mirorred_gaussian' 
        '''

        if self.keep_constant:
            return self,0.
        
        shape = self.value.shape if hasattr(self.value,'shape') else ()
        new_value = self.value + jnp.sqrt(self.proposal_var)*jax.random.normal(key,shape=shape)
        # Mirror values outside bounds component-wise
        if self.min is not None or self.max is not None:
            def mirror_bounds(val):
                if self.min is not None:
                    below_min = val<self.min
                    val = jnp.where(below_min,2*self.min-val,val)
                if self.max is not None:
                    above_max = val>self.max
                    val = jnp.where(above_max,2*self.max-val,val)
                return val, jnp.any(below_min) if self.min is not None else False, jnp.any(above_max) if self.max is not None else False
            
            new_value, below, above = mirror_bounds(new_value)
            while below or above:
                new_value, below, above = mirror_bounds(new_value)
        log_correction = 0.
        return FlatParameter(**{**self.__dict__,'value':new_value}),log_correction

    def update(self, value, accepted): 
        """ Update the parameter with a new value
        
        :param value: The new value to update the parameter with
        :param accepted: Whether the new value was accepted.
        """
        if accepted:
            if self.max and self.min is not None:
                value = jnp.clip(value, self.min, self.max)
            elif self.max is not None:
                value = jnp.clip(value, None, self.max)
            elif self.min is not None:
                value = jnp.clip(value, self.min, None)
            self.value = value
        
    def log_prior(self):
        """ Logarithm of the prior probability of the parameter
        
        :return: The logarithm of the prior probability of the parameter
        """
        return uniform_logpdf(self.value, self.min, self.max).sum()

    def plot_prior(self, x_min=1e-3, x_max=1, n_points=100):
            """Plot the prior distribution of the parameter
            
            :param x_min: Minimum x value to plot
            :param n_points: Number of points to plot
            :param x_max: Maximum x value to plot
            """
            # Create array of parameter values
            x = jnp.linspace(x_min,x_max,n_points)

            # Calculate prior for each value
            prior_vals = [FlatParameter(**{**self.__dict__,'value':val}).log_prior() for val in x]

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
