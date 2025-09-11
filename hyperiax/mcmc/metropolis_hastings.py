import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

def metropolis_hastings(log_target, proposal, data, init, num_samples, burn_in=0, thin=1, rng_key=None, savef=lambda x: x, verbose=True):
    """ MCMC using Metropolis-Hastings algorithm

    :param log_target: Function returning the logarithm of the target distribution
    :param proposal: Function returning a sample from the proposal distribution
    :param data: Observed data
    :param init: Initial state
    :param num_samples: Number of samples to draw
    :param burn_in: Number of samples to discard at the beginning
    :param thin: Thinning rate
    :param rng_key: Random number generator key
    :param savef: Function to save the state
    :param verbose: Whether to show progress bar and acceptance rate
    :return: List of log likelihoods and samples
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    rng_key, subkey = jax.random.split(rng_key)
    current_state,_ = proposal(data, init, subkey)
    log_likelihood_current = log_target(data,current_state)
    log_likelihoods = []
    samples = []
    accepted_count = 0  # Initialize counter for accepted proposals
    accepted_count_params = 0  # Counter for accepted proposals when update_params is True
    accepted_count_no_params = 0  # Counter for accepted proposals when update_params is False
    total_count_params = 0  # Total number of proposals when update_params is True
    total_count_no_params = 0  # Total number of proposals when update_params is False

    iterator=range(num_samples+burn_in)
    if verbose:
        iterator=tqdm(iterator)

    for i in iterator:
        rng_key, subkey = jax.random.split(rng_key)

        # Propose a new state
        update_params = proposal.update_params if hasattr(proposal,'update_params') else False
        update_obs_var = proposal.update_obs_var if hasattr(proposal,'update_obs_var') else False
        proposed_state,log_correction = proposal(data, current_state, subkey)

        # Compute acceptance probability
        log_likelihood_proposed = log_target(data,proposed_state)
        alpha = jnp.minimum(1.0, jnp.exp(log_likelihood_proposed-log_likelihood_current+log_correction))
        if update_params and update_obs_var:
            alpha = 1.

        # Update counters based on update_params
        if update_params:
            total_count_params += 1
        else:
            total_count_no_params += 1

        # Decide whether to accept the proposed state
        if jax.random.uniform(subkey)<alpha or (hasattr(proposal,'accept_next_state') and proposal.accept_next_state):
            current_state = proposed_state
            log_likelihood_current = log_likelihood_proposed
            accepted_count += 1  # Increment overall count
            if update_params:
                accepted_count_params += 1  # Increment params count
            else:
                accepted_count_no_params += 1  # Increment no-params count

        # Save the sample after burn-in and according to thinning rate
        if i >= burn_in and (i - burn_in) % thin == 0:
            log_likelihoods.append(log_likelihood_current)
            samples.append(savef(current_state))

    # acceptance rates
    acceptance_rate = accepted_count/(num_samples+burn_in)
    acceptance_rate_params = accepted_count_params/total_count_params if total_count_params > 0 else 0
    acceptance_rate_no_params = accepted_count_no_params/total_count_no_params if total_count_no_params > 0 else 0
    
    if verbose:
        print(f"Overall acceptance rate: {acceptance_rate:.4f}")
        if hasattr(proposal,'update_params'):
            print(f"Acceptance rate (update_params=True): {acceptance_rate_params:.4f}")
            print(f"Acceptance rate (update_params=False): {acceptance_rate_no_params:.4f}")

    return log_likelihoods,samples