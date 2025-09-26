from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Any, TypeVar

from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax.random import split, uniform, normal, gamma
from tqdm import tqdm

from hyperiax.mcmc.parameters import VarianceParameter

from ..tree import HypTree
from .parameterstore import ParameterStore

MCMCState = TypeVar("MCMCState", ParameterStore, Tuple[ParameterStore, ArrayLike])


##############################
## Abstract Sampler Classes ##
##############################
class StateSampler(ABC):
    def __init__(
        self,
        tree: HypTree,
        params_sampler: ParametersSampler,
        noise_sampler: NoiseSampler,
    ) -> None:
        self.tree = tree
        self.params_sampler = params_sampler
        self.noise_sampler = noise_sampler
        super().__init__()

    @abstractmethod
    def propose_state(
        self, rng_key: Array, current_state: MCMCState, data: ArrayLike
    ) -> Tuple[Any, ArrayLike]:
        """
        Propose a new state given the current state and data.

        Args:
            rng_key (Array): A JAX random key for stochasticity.
            current_state (Any): The current state of the Markov chain.
            data (ArrayLike): The observed data.

        Returns:
            Tuple[Any, ArrayLike]: The proposed new state and the proposal log probability correction.
        """
        ...


class NoiseSampler(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def propose_noise(self, rng_key: Array, current_noise: ArrayLike) -> ArrayLike:
        """
        Propose new noise given the current noise.

        Args:
            rng_key (Array): A JAX random key for stochasticity.
            current_noise (ArrayLike): The current noise.

        Returns:
            ArrayLike: The proposed new noise.
        """
        ...


class ParametersSampler(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def propose_params(
        self, rng_key: Array, current_params: ParameterStore, data: ArrayLike
    ) -> Tuple[ParameterStore, ArrayLike]:
        """
        Propose a new set of parameters given the current parameters.

        Args:
            rng_key (Array): A JAX random key for stochasticity.
            current_params (ParameterStore): The current parameters.

        Returns:
            Tuple[ParameterStore, ArrayLike]: The proposed new parameters and the proposal log probability correction.
        """
        ...


#####################################
## Noise (Wiener Process) Samplers ##
#####################################
class PCNNoiseSampler(NoiseSampler):
    def __init__(self, eta: float) -> None:
        """
        Preconditioned Crank-Nicolson (pCN) sampler for Wiener processes.

        Args:
            eta (float): The step size parameter for the pCN proposal.
        """
        self.eta = eta
        super().__init__()

    def propose_noise(self, rng_key: Array, current_noise: ArrayLike) -> ArrayLike:
        new_noise = normal(
            rng_key, shape=current_noise.shape, dtype=current_noise.dtype
        )
        return self.eta * current_noise + jnp.sqrt(1 - self.eta**2) * new_noise


########################
## Parameter Samplers ##
########################
class MHParametersSampler(ParametersSampler):
    def __init__(self, *args, **kwargs) -> None:
        """
        Standard Metropolis-Hastings sampler for parameters.
        """
        super().__init__()

    def propose_params(
        self, rng_key: Array, current_params: ParameterStore, data: ArrayLike
    ) -> Tuple[ParameterStore, ArrayLike]:
        return current_params.propose(rng_key)


class GibbsParametersSampler(ParametersSampler):
    def __init__(self, tree: HypTree, *args, **kwargs) -> None:
        """
        Gibbs sampler for variance parameters with conjugate priors.

        Args:
            tree (HypTree): The hierarchical tree structure containing the data and parameters.
        """
        self.tree = tree
        super().__init__()

        self.update_obs_var = False  # Start by updating all parameters except obs_var

    def propose_params(
        self, rng_key: Array, current_params: ParameterStore, data: Array
    ) -> Tuple[ParameterStore, ArrayLike]:
        assert current_params["obs_var"].prior == "inv_gamma", (
            "Gibbs sampler only supports inverse gamma prior for obs_var"
        )
        sub_keys = split(rng_key, len(current_params))

        if not self.update_obs_var:  # update all parameters except obs_var
            proposals_and_corrections = {
                k: v.propose(sub_key)
                for sub_key, (k, v) in zip(sub_keys, current_params.params_dict.items())
                if k != "obs_var"
            }
            new_params_dict = {k: v[0] for k, v in proposals_and_corrections.items()}
            new_params_dict["obs_var"] = current_params[
                "obs_var"
            ]  # keep obs_var unchanged
            log_correction = sum(v[1] for v in proposals_and_corrections.values())
        else:  # update only obs_var
            residuals = self._compute_residuals(data)
            new_obs_var = self._sample_obs_var_posterior(
                sub_keys[-1], current_params["obs_var"], residuals
            )

            new_params_dict = {
                k: v for k, v in current_params.params_dict.items() if k != "obs_var"
            }
            new_params_dict["obs_var"] = new_obs_var
            log_correction = jnp.inf  # Make sure to accept this proposal

        self.update_obs_var = not self.update_obs_var  # Alternate for next call

        return ParameterStore(new_params_dict), log_correction

    def _compute_residuals(self, data: ArrayLike) -> ArrayLike:
        """
        Compute residuals between the model's leaf values and the observed data.

        Args:
            data (ArrayLike): The observed data.

        Returns:
            ArrayLike: The residuals.
        """
        return self.tree.data["value"][self.tree.is_leaf] - data

    def _sample_obs_var_posterior(
        self, rng_key: Array, obs_var: VarianceParameter, residuals: ArrayLike
    ) -> VarianceParameter:
        """
        Sample a new observation variance from its posterior distribution (inverse gamma).

        Args:
            rng_key (Array): A JAX random key for stochasticity.
            obs_var_param (VarianceParameter): The current observation variance parameter.
            residuals (ArrayLike): The residuals between model predictions and observed data.

        Returns:
            VarianceParameter: The sampled observation variance parameter.
        """

        if not obs_var.keep_constant:
            alpha_post = (
                obs_var.prior_dist_hparams["alpha"] + 0.5 * residuals.size
            )  # alpha from prior
            beta_post = obs_var.prior_dist_hparams["beta"] + 0.5 * jnp.sum(
                residuals**2
            )  # beta from prior
            new_obs_var_value = beta_post / gamma(
                rng_key, alpha_post, shape=obs_var.value.shape, dtype=residuals.dtype
            )  # inverse gamma sample
            new_obs_var_dict = obs_var.__dict__
            new_obs_var_dict["value"] = new_obs_var_value
            new_obs_var = VarianceParameter(
                **new_obs_var_dict
            )  # create new VarianceParameter instance
        else:
            new_obs_var = obs_var  # keep unchanged if constant

        return new_obs_var


class CanonicalStateSampler(StateSampler):
    def __init__(
        self,
        tree: HypTree,
        params_sampler: ParametersSampler,
        noise_sampler: NoiseSampler,
    ) -> None:
        super().__init__(tree, params_sampler, noise_sampler)

    def propose_state(
        self,
        rng_key: Array,
        current_state: MCMCState,
        data: ArrayLike,
    ) -> Tuple[MCMCState, ArrayLike]:
        params, noise = current_state
        sub_key1, sub_key2 = split(rng_key)
        new_params, log_correction = self.params_sampler.propose_params(
            sub_key1, params, data
        )
        new_noise = self.noise_sampler.propose_noise(sub_key2, noise)
        return (new_params, new_noise), log_correction


class AlternatingStateSampler(StateSampler):
    def __init__(
        self,
        tree: HypTree,
        params_sampler: ParametersSampler,
        noise_sampler: NoiseSampler,
    ) -> None:
        super().__init__(tree, params_sampler, noise_sampler)

        self.update_params = True  # Start by updating parameters

    def propose_state(
        self,
        rng_key: Array,
        current_state: MCMCState,
        data: ArrayLike,
    ) -> Tuple[MCMCState, ArrayLike]:
        params, noise = current_state
        sub_key1, sub_key2 = split(rng_key)

        if self.update_params:
            new_params, log_correction = self.params_sampler.propose_params(
                sub_key1, params, data
            )
            new_noise = noise  # Keep noise unchanged
        else:
            new_noise = self.noise_sampler.propose_noise(sub_key2, noise)
            new_params = params  # Keep parameters unchanged
            log_correction = 0.0  # No correction for noise proposal

        self.update_params = not self.update_params  # Alternate for next call

        return (new_params, new_noise), log_correction


class MetropolisHastingsSampler:
    def __init__(
        self,
        state_sampler: StateSampler,
    ) -> None:
        self.state_sampler = state_sampler
        self.reset_counts()
        super().__init__()

    def reset_counts(self):
        self.accepted_count = 0

    def sample(
        self,
        rng_key: Array,
        log_posterior: Callable[[MCMCState, ArrayLike], ArrayLike],
        log_likelihood: Callable[[ArrayLike, MCMCState], ArrayLike],
        data: ArrayLike,
        init_state: MCMCState,
        num_samples: int,
        num_burn_in: int = 0,
        thinning: int = 1,
        show_progress_bar: bool = True,
    ) -> Tuple[list[ArrayLike], list[Any]]:
        """
        Run the Metropolis-Hastings MCMC sampler.

        Args:
            rng_key (Array): A JAX random key for stochasticity.
            log_posterior (Callable[[MCMCState, ArrayLike], ArrayLike]): Function to compute the log posterior density that we want to sample from. It should accept the current state the observed data, and return the log density.
            log_likelihood (Callable[[ArrayLike, MCMCState], ArrayLike]): Function to compute the log likelihood of the data given the current state.
            data (ArrayLike): The observed data.
            init_state (MCMCState): The initial state of the Markov chain.
            num_samples (int): The number of samples to draw after burn-in.
            num_burn_in (int, optional): The number of burn-in iterations. Defaults to 0.
            thinning (int, optional): The thinning interval. Defaults to 1.

        Returns:
            Tuple[list[ArrayLike], list[Any]]: Lists of log likelihoods and saved samples.
        """
        self.reset_counts()

        rng_key, sub_key = split(rng_key)
        current_state, _ = self.state_sampler.propose_state(sub_key, init_state, data)
        current_log_post = log_posterior(current_state, data)

        log_liki_history = []
        sample_history = []

        pbar = tqdm(
            range(num_samples + num_burn_in),
            desc="MCMC Sampling",
            disable=not show_progress_bar,
        )

        for i in pbar:
            rng_key, sub_key = split(rng_key)

            # Track what type of update we're doing BEFORE the proposal
            # updating_params = self._is_updating_params()
            # gibbs_conditional = self._is_gibbs_conditional()

            # Propose a new state
            proposed_state, log_correction = self.state_sampler.propose_state(
                sub_key, current_state, data
            )

            # Compute acceptance probability
            proposed_log_post = log_posterior(proposed_state, data)
            acceptance_prob = jnp.minimum(
                1.0, jnp.exp(proposed_log_post - current_log_post + log_correction)
            )

            # Accept/reject step
            rng_key, accept_key = split(rng_key)
            accepted = uniform(accept_key) < acceptance_prob
            if accepted:
                current_state = proposed_state
                current_log_post = proposed_log_post
                self.accepted_count += 1

            if not show_progress_bar:
                param_str = ", ".join(
                    f"{k}: {v.value:.3f}" if hasattr(v, "value") and isinstance(v.value, (int, float, float)) else f"{k}: {v}"
                    for k, v in current_state[0].items()
                )
                print(f"Current params: {param_str}, accepted: {accepted}")

            # Save samples after burn-in
            if i >= num_burn_in and (i - num_burn_in) % thinning == 0:
                log_liki_history.append(log_likelihood(data, current_state))
                sample_history.append(current_state)

        # Print acceptance rates
        self._print_acceptance_rates(num_samples + num_burn_in)

        return log_liki_history, sample_history

    def _print_acceptance_rates(self, total_iterations: int):
        """Print acceptance rate statistics."""
        overall_rate = self.accepted_count / total_iterations
        print(f"Overall acceptance rate: {overall_rate:.4f}")


# TODO: Some factory functions for easy migration from old API, might be removed later
def create_mh_sampler(tree: HypTree, eta: float = 0.9) -> MetropolisHastingsSampler:
    """Create MH sampler equivalent to obs_var_sample_conditional=False."""
    param_sampler = MHParametersSampler(tree)
    noise_sampler = PCNNoiseSampler(eta)
    proposal_sampler = AlternatingStateSampler(tree, param_sampler, noise_sampler)
    return MetropolisHastingsSampler(proposal_sampler)


def create_gibbs_sampler(tree: HypTree, eta: float = 0.9) -> MetropolisHastingsSampler:
    """Create Gibbs sampler equivalent to obs_var_sample_conditional=True."""
    param_sampler = GibbsParametersSampler(tree)
    noise_sampler = PCNNoiseSampler(eta)
    proposal_sampler = AlternatingStateSampler(tree, param_sampler, noise_sampler)
    return MetropolisHastingsSampler(proposal_sampler)
