import numpy as np

# Calculate Gelman-Rubin statistics for each parameter
def gelman_rubin(chains):
    """Compute Gelman-Rubin statistic for each parameter
    
    :param chains: List of chains, each containing parameter samples
    """
    n = len(chains[0]) # length of each chain
    m = len(chains) # number of chains
    
    # Get parameter names from first sample of first chain
    param_names = list(chains[0][0].values().keys())
    
    # Convert chains to arrays for each parameter
    chain_arrays = {}
    for param in param_names:
        chain_arrays[param] = np.array([[chain[i][param].value for i in range(n)] for chain in chains])
    
    # Calculate statistics for each parameter
    r_hats = {}
    for param in param_names:
        # Chain means
        chain_means = chain_arrays[param].mean(axis=1)
        # Overall mean
        overall_mean = chain_means.mean()
        
        # Between-chain variance
        B = n*np.var(chain_means,ddof=1)
        
        # Within-chain variance
        W = np.mean([np.var(chain,ddof=1) for chain in chain_arrays[param]])
        
        # Estimated variance
        V = (n-1)/n*W + B/n
        
        # R-hat
        r_hats[param] = np.sqrt(V/W)
    
    return r_hats
