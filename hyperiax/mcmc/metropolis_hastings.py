
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

def metropolis_hastings(log_target, proposal, init, num_samples, burn_in=0, thin=1, rng_key=None, savef=lambda x: x):
    """MCMC using Metropolis-Hastings algorithm."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    current_state = init
    samples = []

    for i in tqdm(range(num_samples + burn_in)):
        rng_key, subkey = jax.random.split(rng_key)

        # Propose a new state
        proposed_state = proposal(current_state, i, subkey)

        # Compute acceptance probability
        alpha = jnp.minimum(1.0, jnp.exp(log_target(proposed_state) - log_target(current_state)))

        # Decide whether to accept the proposed state
        if jax.random.uniform(subkey) < alpha:
            current_state = proposed_state

        # Save the sample after burn-in and according to thinning rate
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(savef(current_state))

    return samples