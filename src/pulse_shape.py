import os
import numpy as np
import matplotlib.pyplot as plt

def estimate_pedestal(X, y, threshold_quantile=0.2):
    """Estimate the per-position pedestal from low-energy windows.

    We select windows whose |y| falls in the lowest quantile (noise-only)
    and compute the mean sample value at each of the 7 positions.

    Parameters
    ----------
    X : np.ndarray [N, 7]
    y : np.ndarray [N]
    threshold_quantile : float
        Fraction of lowest |y| values to use.

    Returns
    -------
    pedestal : np.ndarray [7]
    """
    abs_y = np.abs(y)
    cutoff = np.quantile(abs_y, threshold_quantile)
    mask = abs_y <= cutoff
    pedestal = X[mask].mean(axis=0)

    return pedestal


def estimate_template(X, y, top_percentile=90):
    """Estimate the average pulse template from high-energy windows.

    Selects windows where y is in the top percentile (clearly signal-dominated)
    and computes the mean shape, then normalizes to unit peak.

    Parameters
    ----------
    X : np.ndarray [N, 7]
    y : np.ndarray [N]
    top_percentile : float
        Percentile threshold (e.g. 90 means top 10% by energy).

    Returns
    -------
    g : np.ndarray [7]
        Normalized pulse template (peak = 1.0).
    """
    cutoff = np.percentile(y, top_percentile)
    mask = y >= cutoff
    g_raw = X[mask].mean(axis=0)
    baseline = g_raw.min()
    g_shifted = g_raw - baseline
    peak = g_shifted.max()
    if peak > 0:
        g = g_shifted / peak

    else:
        # Fallback: return raw mean (should not happen with real data)
        g = g_raw / (g_raw.max() if g_raw.max() > 0 else 1.0)

    return g


def estimate_noise_covariance(X, y, threshold_quantile=0.2):
    """Estimate the 7x7 noise covariance from low-energy (noise-only) windows.

    Parameters
    ----------
    X : np.ndarray [N, 7]
    y : np.ndarray [N]
    threshold_quantile : float
        Fraction of lowest |y| values to use.

    Returns
    -------
    C : np.ndarray [7, 7]
    """
    abs_y = np.abs(y)
    cutoff = np.quantile(abs_y, threshold_quantile)
    mask = abs_y <= cutoff
    X_noise = X[mask]

    # Subtract mean (pedestal) before computing covariance
    C = np.cov(X_noise, rowvar=False)
    
    return C


def plot_template(g, save_dir):
    """Bar plot of the estimated 7-sample pulse template."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(g))
    bars = ax.bar(positions, g, color="#6366f1", edgecolor="white", alpha=0.9)

    for bar, val in zip(bars, g):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Sample position in window")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Estimated pulse template g (from high-energy windows)")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"s{i}" for i in positions])
    ax.set_ylim(bottom=-0.05, top=1.25)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "pulse_template.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'pulse_template.png')}")


def plot_covariance(C, save_dir):
    """Heatmap of the 7x7 noise covariance matrix."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="coolwarm", aspect="equal")
    ax.set_xticks(range(C.shape[0]))
    ax.set_yticks(range(C.shape[1]))
    ax.set_xticklabels([f"s{i}" for i in range(C.shape[0])])
    ax.set_yticklabels([f"s{i}" for i in range(C.shape[1])])
    ax.set_title("Noise covariance matrix C (from low-energy windows)")

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C[i, j]:.2e}", ha="center", va="center",
                    fontsize=7, color="white" if abs(C[i, j]) > 0.5 * np.max(np.abs(C)) else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "noise_covariance.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'noise_covariance.png')}")


def run_pulse_analysis(X, y, save_dir):
    """Run full pulse shape analysis and return (g, C, pedestal).

    Parameters
    ----------
    X : np.ndarray [N, 7]
    y : np.ndarray [N]
    save_dir : str

    Returns
    -------
    g : np.ndarray [7]     -- pulse template
    C : np.ndarray [7, 7]  -- noise covariance
    pedestal : np.ndarray [7]
    """
    print("\n--- Pulse shape analysis ---")
    pedestal = estimate_pedestal(X, y)
    print(f"  Pedestal (per-position): {pedestal}")
    g = estimate_template(X, y)
    print(f"  Template g: {g}")
    C = estimate_noise_covariance(X, y)
    print(f"  Noise covariance C diagonal: {np.diag(C)}")
    figs_dir = os.path.join(save_dir, "pulse_analysis")
    plot_template(g, figs_dir)
    plot_covariance(C, figs_dir)
    print("--- Pulse shape analysis complete ---\n")

    return g, C, pedestal
