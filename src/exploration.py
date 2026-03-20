import os
import numpy as np
import matplotlib.pyplot as plt

def print_data_summary(data, split_name=""):
    """Print basic statistics for a loaded split."""
    prefix = f"[{split_name}] " if split_name else ""
    X = data["X"]
    y = data["y"]
    y_OF = data["y_OF"]
    print(f"{prefix}X (lo-gain samples):  shape={X.shape}, "
          f"dtype={X.dtype}, min={X.min():.4f}, max={X.max():.4f}, "
          f"mean={X.mean():.4f}")
    
    print(f"{prefix}y (lo-gain target):   shape={y.shape}, "
          f"dtype={y.dtype}, min={y.min():.4f}, max={y.max():.4f}, "
          f"mean={y.mean():.4f}")
    
    print(f"{prefix}y_OF (OF baseline):   shape={y_OF.shape}, "
          f"dtype={y_OF.dtype}, min={y_OF.min():.4f}, max={y_OF.max():.4f}, "
          f"mean={y_OF.mean():.4f}")
    
    print()


def plot_sample_histograms(X, save_dir, n_bins=200):
    """Histogram of all sample values across the 7-sample windows.

    This reveals the pedestal distribution, noise width, and the tail
    from pulse deposits.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    flat = X.flatten()
    axes[0].hist(flat, bins=n_bins, color="#3b82f6", edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("sample_lo value (normalized)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Global sample_lo distribution")
    axes[0].set_yscale("log")

    # Per-position histogram (stacked)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, X.shape[1]))

    for i in range(X.shape[1]):
        axes[1].hist(X[:, i], bins=n_bins, alpha=0.5, label=f"pos {i}",
                     color=colors[i], edgecolor="none")
        
    axes[1].set_xlabel("sample_lo value (normalized)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("sample_lo distribution by window position")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "sample_lo_histograms.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'sample_lo_histograms.png')}")


def plot_energy_histogram(y, y_phys, save_dir, n_bins=200):
    """Histogram of target energies in both normalized and physical units."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(y, bins=n_bins, color="#10b981", edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("ene_lo (normalized)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Target energy distribution (normalized)")
    axes[0].set_yscale("log")
    axes[1].hist(y_phys, bins=n_bins, color="#f59e0b", edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("ene_lo (physical units)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Target energy distribution (physical)")
    axes[1].set_yscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "energy_histograms.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'energy_histograms.png')}")


def plot_sample_windows(X, y, save_dir, n_examples=16):
    """Plot a grid of individual 7-sample windows colored by target energy.

    Helps visualize pulse shapes and how they relate to the target.
    """
    os.makedirs(save_dir, exist_ok=True)
    sorted_idx = np.argsort(y)
    step = max(1, len(sorted_idx) // n_examples)
    indices = sorted_idx[::step][:n_examples]

    ncols = 4
    nrows = (n_examples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), sharex=True)
    axes = axes.flatten()

    positions = np.arange(X.shape[1])

    for ax_idx, data_idx in enumerate(indices):
        ax = axes[ax_idx]
        ax.bar(positions, X[data_idx], color="#6366f1", edgecolor="none", alpha=0.8)
        ax.set_title(f"y={y[data_idx]:.3f}", fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"s{i}" for i in positions], fontsize=7)

    for ax_idx in range(len(indices), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Sample windows (sorted by target energy)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "sample_windows_grid.png"),
                dpi=150, bbox_inches="tight")
    
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'sample_windows_grid.png')}")


def plot_correlation_matrix(X, save_dir):
    """Correlation matrix between the 7 sample positions."""
    os.makedirs(save_dir, exist_ok=True)
    corr = np.corrcoef(X.T)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(X.shape[1]))
    ax.set_yticks(range(X.shape[1]))
    ax.set_xticklabels([f"s{i}" for i in range(X.shape[1])])
    ax.set_yticklabels([f"s{i}" for i in range(X.shape[1])])
    ax.set_title("Sample position correlation matrix")

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr[i, j]) > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "sample_correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'sample_correlation_matrix.png')}")


def run_exploration(data, y_phys, save_dir, split_name="train"):
    """Run all exploratory diagnostics."""
    print(f"\n--- Exploration: {split_name} ---")
    print_data_summary(data, split_name)

    figs_dir = os.path.join(save_dir, "exploration")
    plot_sample_histograms(data["X"], figs_dir)
    plot_energy_histogram(data["y"], y_phys, figs_dir)
    plot_sample_windows(data["X"], data["y"], figs_dir)
    plot_correlation_matrix(data["X"], figs_dir)
    print(f"--- Exploration complete ---\n")
