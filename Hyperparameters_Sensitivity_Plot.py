import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

def plot_feature_std_bar_from_var(feature_names, variances, model_name="", title="Hyperparameters Sensitivity via Std", save_path=None):
    """
    Plots a bar chart of standard deviations (sqrt of variance) per feature,
    with darker color for higher std.

    Args:
        feature_names (list of str): Hyperparameter names.
        variances (list of float): Corresponding variance values.
        model_name (str): Model name to display.
        title (str): Plot title.
        save_path (str or None): Path to save the plot.
    """
    variances = np.array(variances)
    stds = np.sqrt(variances)

    # Sort features by std descending
    sorted_indices = np.argsort(stds)[::-1]
    sorted_features = np.array(feature_names)[sorted_indices]
    sorted_stds = stds[sorted_indices]

    cmap = cm.magma_r  # Reverse copper so darker = higher
    norm = Normalize(vmin=min(sorted_stds), vmax=max(sorted_stds))
    colors = [cmap(norm(val)) for val in sorted_stds]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(sorted_features)), sorted_stds, color=colors)
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha='right')
    ax.set_ylabel("Standard Deviation")
    ax.set_title(f"{title} - {model_name}" if model_name else title)

    # Bar labels
    for bar, val in zip(bars, sorted_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}",
                ha='center', va='bottom', fontsize=8)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Standard Deviation Scale ")

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
plot_feature_variance_bar(feature_name,var,model_name="TimesNet" ,save_path="plot_var_timesnet",log_scale=False)

plot_feature_std_bar_from_var(feature_name, var, model_name="TimesNet", save_path="std_plot_timesnet.png")

