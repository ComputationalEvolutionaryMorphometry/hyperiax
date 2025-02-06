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

    iterator=range(num_samples+burn_in)
    if verbose:
        iterator=tqdm(iterator)

    for i in iterator:
        rng_key, subkey = jax.random.split(rng_key)

        # Propose a new state
        proposed_state,log_correction = proposal(data, current_state, subkey)

        # Compute acceptance probability
        log_likelihood_proposed = log_target(data,proposed_state)
        alpha = jnp.minimum(1.0, jnp.exp(log_likelihood_proposed-log_likelihood_current+log_correction))
        #print(alpha,log_likelihood_proposed,log_likelihood_current,proposed_state[0].values(),current_state[0].values())

        # Decide whether to accept the proposed state
        if jax.random.uniform(subkey) < alpha:
            current_state = proposed_state
            log_likelihood_current = log_likelihood_proposed
            accepted_count += 1  # Increment count if proposal is accepted

        # Save the sample after burn-in and according to thinning rate
        if i >= burn_in and (i - burn_in) % thin == 0:
            log_likelihoods.append(log_likelihood_current)
            samples.append(savef(current_state))

    # acceptance rate
    acceptance_rate=accepted_count/(num_samples+burn_in)
    if verbose:
        print(f"Acceptance rate: {acceptance_rate:.4f}")

    return log_likelihoods,samples