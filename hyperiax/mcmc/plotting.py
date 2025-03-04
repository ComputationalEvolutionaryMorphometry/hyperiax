import numpy as np
from matplotlib import pyplot as plt

# trace plots for MCMC
def trace_plots(samples,true_params=None,rewrite_names=None):
    """ Trace plots 
    
    :param samples: A list of ParameterStore objects.
    """

    # Determine grid layout
    num_params = len(samples[0])
    num_rows = int(np.sqrt(num_params))
    num_cols = num_params // num_rows + (num_params % num_rows > 0)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5))
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D array

    # Plot traces
    for ax, (i, param) in zip(axes.ravel(), enumerate(samples[0].values().keys())):
        name = param if rewrite_names is None else rewrite_names[param]
        ax.plot(np.array([sample[param].value for sample in samples]))
        ax.set_title(f"Trace for {name}")
        ax.set_xlabel('Iteration')
        # Add mean line
        mean_val = np.mean([sample[param].value for sample in samples])
        ax.axhline(y=mean_val,color='r',linestyle='-')
        ax.set_ylabel(name)
        ax.grid(True)
        if true_params is not None:
            ax.axhline(y=true_params[param],color='g',linestyle='--')
        ax.set_ylim(bottom=0)

    # Remove any unused subplots
    for ax in axes.ravel()[num_params:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()