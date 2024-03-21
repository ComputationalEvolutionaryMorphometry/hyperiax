import numpy as np
from matplotlib import pyplot as plt

# trace plots for MCMC
def trace_plots(samples):
    """ Trace plots """

    # Determine grid layout
    num_params = len(samples[0])
    num_rows = int(np.sqrt(num_params))
    num_cols = num_params // num_rows + (num_params % num_rows > 0)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5))
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D array

    # Plot traces
    for ax, (i, param) in zip(axes.ravel(), enumerate(samples[0].values().keys())):
        ax.plot(np.array([sample[param].value for sample in samples]))
        ax.set_title(f"Trace for {param}")
        ax.set_xlabel('Iteration')
        ax.set_ylabel(param)
        ax.grid(True)

    # Remove any unused subplots
    for ax in axes.ravel()[num_params:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()