import numpy as np
from matplotlib import pyplot as plt

# trace plots for MCMC
def trace_plots(samples,true_params=None,rewrite_names=None,y_bottom=0.):
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
        values = np.array([sample[param].value for sample in samples])
        if values.ndim > 1:
            # For multidimensional values, compute mean for each dimension
            mean_vals = np.mean(values, axis=0)
            for i, mean_val in enumerate(mean_vals):
                ax.axhline(y=mean_val, color='r', linestyle='-', alpha=0.7, label=f'Mean dim {i}' if i==0 else None)
        else:
            # For scalar values, just one mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='r', linestyle='-')
        ax.set_ylabel(name)
        ax.grid(True)
        if true_params is not None:
            # Handle both scalar and array true parameters
            true_param_values = np.atleast_1d(true_params[param])
            for i, true_val in enumerate(true_param_values):
                ax.axhline(y=true_val, color='g', linestyle='--', label=f'True value dim {i}' if i==0 and len(true_param_values)>1 else None)
        ax.set_ylim(bottom=y_bottom)

    # Remove any unused subplots
    for ax in axes.ravel()[num_params:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()