from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Literal

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from jax.random import normal

from .utils import inverse_gamma_logpdf, uniform_logpdf


class Parameter(ABC):
    """
    Base class for a parameter in a Markov Chain Monte Carlo simulation.

    Attributes:
        value (ArrayLike): The current value of the parameter.

    Methods:
        propose(rng_key: Array) -> Tuple[Parameter, ArrayLike]: Propose a new value for the parameter.
        log_prior() -> ArrayLike: Logarithm of the prior probability of the current parameter.
    """

    def __init__(self, value: ArrayLike, dtype: DTypeLike = jnp.float32) -> None:
        self.value = jnp.asarray(value, dtype=dtype)
        self.dtype = dtype

    @abstractmethod
    def propose(self, rng_key: Array) -> Tuple[Parameter, ArrayLike]:
        """
        Propose a new value for the parameter.

        Args:
            key (Array): The key to generate the proposal.

        Returns:
            Tuple[Parameter, ArrayLike]: The proposed parameter and the log pdf correction of the proposal if the proposal is asymmetric.
        """
        ...

    @abstractmethod
    def log_prior(self) -> ArrayLike:
        """
        Logarithm of the prior probability of the current parameter.

        Returns:
            ArrayLike: The log prior probability.
        """
        ...


class FixedParameter(Parameter):
    """
    A parameter that is constant regardless of updates.  The parameter always have the value passed in the constructor

    Attributes:
        value (ArrayLike): The fixed value of the parameter.

    Methods:
        propose(rng_key: Array) -> FixedParameter: Propose a new value for the parameter, which is always the same as the current value.
        update(value: ArrayLike, accepted: bool) -> None: Update the parameter, which does nothing as the parameter is fixed.
        log_prior() -> ArrayLike: Logarithm of the prior probability of the parameter, which is always 0.
    """

    def __init__(self, value: ArrayLike, dtype: DTypeLike = jnp.float32) -> None:
        super().__init__(value, dtype)

    def propose(self, rng_key: Array) -> Tuple[FixedParameter, ArrayLike]:
        """
        Propose a new value for the parameter

        Args:
            rng_key (Array): A key to sample with

        Returns:
            Tuple[FixedParameter, float]: The proposed parameter (which is the same as the current parameter) and 0.0 as the log correction term.
        """
        return FixedParameter(self.value, self.dtype), 0.0

    def log_prior(self) -> ArrayLike:
        """
        Logarithm of the prior probability of the parameter

        Returns:
            0. as the parameter is fixed
        """
        return 0.0


class VarianceParameter(Parameter):
    """
    A variance parameter that can be updated during MCMC sampling.

    Attributes:
        value (ArrayLike): The current value of the parameter.
        proposal (Literal['log_normal', 'mirrored_gaussian']): The proposal distribution for the parameter.
        proposal_dist_hparams (dict): Hyperparameters for the proposal distribution.
        proposal_var (ArrayLike): The variance of the proposal distribution.
        prior (Literal['inv_gamma', 'uniform']): The prior distribution for the parameter.
        prior_dist_hparams (dict): Hyperparameters for the prior distribution.
        keep_constant (bool): Whether this parameter keeps constant.

    Methods:
        propose(rng_key: Array) -> Tuple[VarianceParameter, float]: Propose a new value for the parameter.
        update(value: ArrayLike, accepted: bool) -> None: Update the parameter's value.
        log_prior() -> Array: Logarithm of the prior probability of the current parameter.
    """

    def __init__(
        self,
        value: ArrayLike,
        proposal: Literal["log_normal", "mirrored_gaussian"] = "log_normal",
        proposal_dist_hparams: dict = {"min": None, "max": None},
        proposal_var: ArrayLike = 0.01,
        prior: Literal["inv_gamma", "uniform"] = "inv_gamma",
        prior_dist_hparams: dict = {"alpha": 3.0, "beta": 2.0},
        keep_constant: bool = False,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        """
        VarianceParameter, which is a subclass of `mcmc.Parameter`, specify a parameter that can be updated during MCMC sampling.

        Args:
            value (ArrayLike): The initial value of the parameter.
            proposal (Literal['log_normal', 'mirrored_gaussian'], optional): The proposal distribution for the parameter. Defaults to 'log_normal'.
            proposal_dist_hparams (dict, optional): Hyperparameters for the proposal distribution. For 'log_normal', specify 'min' and 'max' bounds. For 'mirrored_gaussian', specify 'min' and 'max' bounds. Defaults to {'min': None, 'max': None}.
            proposal_var (ArrayLike, optional): The variance of the proposal distribution. Defaults to 0.01.
            prior (Literal['inv_gamma', 'uniform'], optional): The prior distribution for the parameter. Defaults to 'inv_gamma'.
            prior_dist_hparams (dict, optional): Hyperparameters for the prior distribution. For 'inv_gamma', specify 'alpha' and 'beta'. For 'uniform', specify 'min' and 'max'. Defaults to {'alpha': 3.0, 'beta': 2.0}.
            keep_constant (bool, optional): Whether this parameter keeps constant. Defaults to False.
        """
        super().__init__(value, dtype)
        self.proposal = proposal
        self.proposal_dist_hparams = proposal_dist_hparams
        self.proposal_var = proposal_var
        self.prior = prior
        self.prior_dist_hparams = prior_dist_hparams
        self.keep_constant = keep_constant

    def propose(self, rng_key: Array) -> Tuple[VarianceParameter, float]:
        """
        Propose a new value for the parameter using the specified proposal distribution.

        Args:
            rng_key (Array): The random key to use for sampling.

        Raises:
            ValueError: If the proposal distribution is unknown.

        Returns:
            Tuple[VarianceParameter, float]: The proposed parameter and the log correction term if the proposal distribution is asymmetric.
        """

        if self.keep_constant:
            return VarianceParameter(**self.__dict__), 0.0

        if self.proposal == "log_normal":
            assert (
                "min" in self.proposal_dist_hparams
                and "max" in self.proposal_dist_hparams
            ), "min and max must be specified for log-normal proposal"
            min_val = self.proposal_dist_hparams["min"]
            max_val = self.proposal_dist_hparams["max"]

            shape = self.value.shape
            noise = jnp.sqrt(self.proposal_var) * normal(
                rng_key, shape=shape, dtype=self.dtype
            )
            new_value = jnp.exp(jnp.log(self.value) + noise)

            # Enforce bounds
            if min_val is not None:
                new_value = jnp.clip(new_value, min_val, jnp.inf)
            if max_val is not None:
                new_value = jnp.clip(new_value, 0, max_val)

            # Add correction term for log-normal proposal asymmetry
            log_correction = (
                -noise
            )  # Since q(x'|x)/q(x|x') = exp(-eps) where eps is the noise
            return VarianceParameter(
                **{**self.__dict__, "value": new_value}
            ), log_correction

        elif self.proposal == "mirrored_gaussian":
            assert (
                "min" in self.proposal_dist_hparams
                and "max" in self.proposal_dist_hparams
            ), "min and max must be specified for mirrored gaussian proposal"
            min_val = self.proposal_dist_hparams["min"]
            max_val = self.proposal_dist_hparams["max"]

            shape = self.value.shape
            noise = jnp.sqrt(self.proposal_var) * normal(
                rng_key, shape=shape, dtype=self.dtype
            )
            new_value = self.value + noise
            # Mirror values outside bounds component-wise
            if min_val is not None:
                below_min = new_value < min_val
                new_value = jnp.where(below_min, 2 * min_val - new_value, new_value)
            if max_val is not None:
                above_max = new_value > max_val
                new_value = jnp.where(above_max, 2 * max_val - new_value, new_value)
            # No correction term for mirrored gaussian proposal since it is symmetric
            log_correction = 0.0
            return VarianceParameter(
                **{**self.__dict__, "value": new_value}
            ), log_correction
        else:
            raise ValueError(f"Unknown proposal type: {self.proposal}")

    def log_prior(self) -> Array:
        """
        Compute the logarithm of the prior probability of the current parameter.

        Raises:
            ValueError: If the prior distribution is unknown.

        Returns:
            Array: The logarithm of the prior probability.
        """
        if self.prior == "inv_gamma":
            assert "alpha" in self.prior_dist_hparams, (
                "alpha must be specified for inv_gamma prior"
            )
            assert "beta" in self.prior_dist_hparams, (
                "beta must be specified for inv_gamma prior"
            )
            alpha = self.prior_dist_hparams["alpha"]
            beta = self.prior_dist_hparams["beta"]
            return inverse_gamma_logpdf(self.value, alpha, beta).sum()
        elif self.prior == "uniform":
            assert (
                "min" in self.prior_dist_hparams and "max" in self.prior_dist_hparams
            ), "min and max must be specified for uniform prior"
            min_val = self.prior_dist_hparams["min"]
            max_val = self.prior_dist_hparams["max"]
            return uniform_logpdf(self.value, min_val, max_val).sum()
        else:
            raise ValueError(f"Unknown prior type: {self.prior}")

    # ?: Consider removing this plotting function to keep the Parameter class lightweight
    # def plot_prior(self, x_min=1e-3, x_max=1, n_points=100):
    #     """Plot the prior distribution of the parameter

    #     :param x_min: Minimum x value to plot
    #     :param n_points: Number of points to plot
    #     :param x_max: Maximum x value to plot
    #     """
    #     # Create array of parameter values
    #     x = jnp.linspace(x_min, x_max, n_points)

    #     # Calculate prior for each value
    #     prior_vals = [
    #         VarianceParameter(**{**self.__dict__, "value": val}).log_prior()
    #         for val in x
    #     ]

    #     # Plot exp(log_prior)
    #     plt.figure()
    #     plt.plot(x, jnp.exp(jnp.array(prior_vals)))
    #     plt.xlabel("Parameter value")
    #     plt.ylabel("Prior density")
    #     plt.title("Prior distribution")

    #     # Find x value where prior is maximized
    #     max_idx = jnp.argmax(jnp.exp(jnp.array(prior_vals)))
    #     max_x = x[max_idx]
    #     print(f"Prior is maximized at x = {max_x}")


class FlatParameter(VarianceParameter):
    """
    A variance parameter with a uniform prior and a mirrored gaussian proposal distribution.
    """

    def __init__(
        self,
        value: ArrayLike,
        proposal_dist_hparams: dict = {"min": None, "max": None},
        proposal_var: ArrayLike = 0.01,
        keep_constant: bool = False,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        super().__init__(
            value=value,
            proposal="mirrored_gaussian",
            proposal_dist_hparams=proposal_dist_hparams,
            proposal_var=proposal_var,
            prior="uniform",
            prior_dist_hparams={
                "min": proposal_dist_hparams["min"],
                "max": proposal_dist_hparams["max"],
            },
            keep_constant=keep_constant,
            dtype=dtype,
        )
