from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.scipy.stats import uniform


def inverse_gamma_logpdf(x: ArrayLike, alpha: ArrayLike, beta: ArrayLike) -> Array:
    """
    Log PDF of the inverse gamma distribution:
    log p(x) = alpha * log(beta) - gammaln(alpha) - (alpha + 1) * log(x) - beta / x

    Args:
        x (ArrayLike): The value to evaluate the PDF at
        alpha (ArrayLike): The alpha parameter of the inverse gamma distribution
        beta (ArrayLike): The beta parameter of the inverse gamma distribution

    Returns:
        Array: The logarithm of the PDF of the inverse gamma distribution at x
    """
    return alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1) * jnp.log(x) - beta / x


def uniform_logpdf(x: ArrayLike, a: ArrayLike, b: ArrayLike) -> Array:
    """
    Log PDF of the uniform distribution:
    log p(x) = -log(b - a), if a <= x <= b
             = -inf,        otherwise

    Args:
        x (ArrayLike): The value to evaluate the PDF at
        a (ArrayLike): The left boundary of the uniform distribution
        b (ArrayLike): The right boundary of the uniform distribution

    Returns:
        Array: The logarithm of the PDF of the uniform distribution at x
    """
    return uniform.logpdf(x, loc=a, scale=b - a)
