import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

def plot_feature_variance_bar(feature_names, variances, model_name="", title="Hyperparameters Sensitivity Analysis via Variance", save_path=None, log_scale=True):
    variances = np.array(variances)
    
    # Optionally apply log scale to variance values
    if log_scale:
        log_variances = np.log10(variances + 1e-10)  # avoid log(0) + add constant 
    else:
        log_variances = variances

    # Invert values so that higher variance (closer to 0 in log) appears darker
    color_vals = -log_variances

    # Set color map and normalize color scale
    cmap = cm.copper
    norm = Normalize(vmin=min(color_vals), vmax=max(color_vals))
    colors = [cmap(norm(v)) for v in color_vals]

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(feature_names)), log_variances, color=colors)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel("Log Variance" if log_scale else "Variance")
    ax.set_title(f"{title + ' - ' + model_name}" if model_name else title)

    # Create reversed colorbar (darker at the top)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.invert_yaxis()
    cbar.set_label("Color ∝ -Log Variance" if log_scale else "Color ∝ -Variance")

    # Customize colorbar ticks to show negative log values (i.e., actual log variances)
    ticks = np.linspace(min(color_vals), max(color_vals), num=5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{-t:.0f}" for t in ticks])  # Flip sign back for display

    # Save or show the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# Example usage:
# plot_feature_variance_bar(["batch_size", "lr", "hidden_dim"], [0.003, 0.007, 0.001])
feature_name=['batch_size','n_layers','top_k','d_model'	,'d_ffn_ratio',	'n_kernels',	'dropout',	'lr','window size']
var = [
3.52E-06,
0.000153196,
3.35E-05,
3.47E-05,
3.09E-06,
0.119547688,
8.60E-06,
0.000122332,
8.08E-05
]
plot_feature_variance_bar(feature_name,var,model_name="TimesNet" ,save_path="plot_var_timesnet")
