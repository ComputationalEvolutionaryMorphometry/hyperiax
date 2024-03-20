
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

def metropolis_hastings(log_target, proposal, data, init, num_samples, burn_in=0, thin=1, rng_key=None, savef=lambda x: x):
    """MCMC using Metropolis-Hastings algorithm."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    current_state = init
    log_likelihood_current = log_target(data,current_state)
    log_likelihoods = []
    samples = []
    accepted_count = 0  # Initialize counter for accepted proposals

    for i in tqdm(range(num_samples + burn_in)):
        rng_key, subkey = jax.random.split(rng_key)

        # Propose a new state
        proposed_state = proposal(data, current_state, subkey)

        # Compute acceptance probability
        log_likelihood_proposed = log_target(data,proposed_state)
        alpha = jnp.minimum(1.0, jnp.exp(log_likelihood_proposed - log_likelihood_current))
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
    acceptance_rate = accepted_count / (num_samples + burn_in)
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    return log_likelihoods,samples