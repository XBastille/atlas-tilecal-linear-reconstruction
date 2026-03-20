import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.io import denormalize
from scipy.stats import norm

def compute_relative_error(y_hat, y, min_energy=10.0):
    """Compute relative error r = (y_hat - y) / y with near-zero protection.

    Formula:  r_i = (y_hat_i - y_i) / y_i

    Only events with y_i >= min_energy are included.  All physical
    energies in this dataset are non-negative (min ~0.25 MeV), so
    there is no need for an absolute-value wrapper.

    Parameters
    ----------
    y_hat : np.ndarray [N]  -- predictions (physical units)
    y : np.ndarray [N]      -- true targets (physical units)
    min_energy : float      -- threshold on y (physical units, e.g. MeV)

    Returns
    -------
    r : np.ndarray [M]          -- relative error (M <= N)
    y_selected : np.ndarray [M] -- corresponding true energies
    mask : np.ndarray [N] bool  -- which events were included
    """
    mask = y >= min_energy
    y_sel = y[mask]
    y_hat_sel = y_hat[mask]
    r = (y_hat_sel - y_sel) / y_sel

    return r, y_sel, mask


def compute_summary_stats(r):
    """Compute summary statistics from relative error array.

    Returns
    -------
    dict with keys: mean, rms, std, median, n_samples
    """
    return {
        "mean": float(np.mean(r)),
        "rms": float(np.sqrt(np.mean(r ** 2))),
        "std": float(np.std(r)),
        "median": float(np.median(r)),
        "n_samples": int(len(r)),
    }


def evaluate_estimator(y_hat_norm, y_norm, y_stats, method_name, figs_dir,
                       make_plots=True, min_energy_phys=10.0):
    """Evaluate a linear estimator: denormalize, compute relative error, plot.

    Parameters
    ----------
    y_hat_norm : np.ndarray [N]  -- predictions in normalized units
    y_norm : np.ndarray [N]      -- targets in normalized units
    y_stats : dict               -- normalization stats (mean_lo, std_lo)
    method_name : str
    figs_dir : str
    make_plots : bool
    min_energy_phys : float      -- threshold in physical units

    Returns
    -------
    stats : dict with mean, rms, std, median, n_samples
    r : np.ndarray               -- relative errors
    y_sel : np.ndarray           -- selected physical targets
    """
    y_hat_phys = denormalize(y_hat_norm, y_stats["mean_lo"], y_stats["std_lo"])
    y_phys = denormalize(y_norm, y_stats["mean_lo"], y_stats["std_lo"])
    r, y_sel, mask = compute_relative_error(y_hat_phys, y_phys, min_energy_phys)
    stats = compute_summary_stats(r)
    print(f"\n  [{method_name}] Mean = {stats['mean']:.6f}")
    print(f"  [{method_name}] RMS  = {stats['rms']:.6f}")
    print(f"  [{method_name}] Std  = {stats['std']:.6f}")
    print(f"  [{method_name}] N    = {stats['n_samples']} "
          f"(excluded {(~mask).sum()} with E < {min_energy_phys})")

    if make_plots:
        plot_relative_error_hist(r, stats, figs_dir, label=method_name)
        plot_relative_error_vs_energy(r, y_sel, figs_dir, label=method_name)

    return stats, r, y_sel

def threshold_stability_study(y_hat, y, thresholds=None, save_dir=None):
    """Sweep over multiple y thresholds and report metrics for each.

    This table demonstrates that the chosen threshold is defensible:
    the Mean and RMS are stable across a range of reasonable thresholds,
    ruling out the possibility that the result is an artifact of one
    cherry-picked cut value.

    Parameters
    ----------
    y_hat : np.ndarray [N]    -- predictions (physical units)
    y : np.ndarray [N]        -- true targets (physical units)
    thresholds : list of float -- y thresholds to evaluate
    save_dir : str or None     -- if provided, save table as PNG

    Returns
    -------
    rows : list of dict  -- one per threshold
    """
    if thresholds is None:
        thresholds = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    total = len(y)
    rows = []
    print("\n  Threshold stability study:")
    print(f"  {'Threshold':>12s}  {'N':>8s}  {'N_excluded':>12s}  "
          f"{'%_kept':>8s}  {'Mean':>10s}  {'RMS':>10s}  {'Std':>10s}")
    
    print("  " + "-" * 82)

    for t in thresholds:
        r, y_sel, mask = compute_relative_error(y_hat, y, t)
        if len(r) == 0:
            continue

        stats = compute_summary_stats(r)
        n_excl = int((~mask).sum())
        pct_kept = 100.0 * stats["n_samples"] / total

        row = {
            "threshold": t,
            "n_samples": stats["n_samples"],
            "n_excluded": n_excl,
            "pct_kept": pct_kept,
            "mean": stats["mean"],
            "rms": stats["rms"],
            "std": stats["std"],
        }

        rows.append(row)

        print(f"  {t:12.1f}  {stats['n_samples']:8d}  {n_excl:12d}  "
              f"{pct_kept:7.2f}%  {stats['mean']:10.6f}  "
              f"{stats['rms']:10.6f}  {stats['std']:10.6f}")

    if save_dir is not None and len(rows) > 0:
        os.makedirs(save_dir, exist_ok=True)
        _plot_stability_table(rows, save_dir)
        _plot_stability_curves(rows, save_dir)

    return rows


def _plot_stability_table(rows, save_dir):
    """Render the stability table as a PNG figure for the report."""
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.6))
    ax.axis("off")
    col_labels = ["y threshold\n(MeV)", "N", "N excluded",
                  "% kept", "Mean", "RMS", "Std"]
    
    cell_text = []

    for r in rows:
        cell_text.append([
            f"{r['threshold']:.0f}",
            f"{r['n_samples']}",
            f"{r['n_excluded']}",
            f"{r['pct_kept']:.2f}%",
            f"{r['mean']:.6f}",
            f"{r['rms']:.6f}",
            f"{r['std']:.6f}",
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Highlight the 10 MeV row (our chosen threshold)
    for j in range(len(col_labels)):
        for i, r in enumerate(rows):
            if r["threshold"] == 10.0:
                table[i + 1, j].set_facecolor("#dbeafe")

    ax.set_title("Threshold stability study -- relative error metrics\n"
                 "(blue row = chosen threshold)",
                 fontsize=12, pad=12)

    plt.tight_layout()
    path = os.path.join(save_dir, "threshold_stability_table.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_stability_curves(rows, save_dir):
    """Plot Mean and RMS as a function of threshold."""
    thresholds = [r["threshold"] for r in rows]
    means = [r["mean"] for r in rows]
    rms_vals = [r["rms"] for r in rows]
    n_vals = [r["n_samples"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(thresholds, means, "o-", color="#3b82f6", linewidth=2)
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axvline(x=10.0, color="#ef4444", linestyle=":", linewidth=1.5,
                    label="chosen = 10 MeV")
    
    axes[0].set_xlabel("y threshold (MeV)")
    axes[0].set_ylabel("Mean relative error")
    axes[0].set_title("Bias stability")
    axes[0].legend()
    axes[1].plot(thresholds, rms_vals, "o-", color="#f59e0b", linewidth=2)
    axes[1].axvline(x=10.0, color="#ef4444", linestyle=":", linewidth=1.5,
                    label="chosen = 10 MeV")
    
    axes[1].set_xlabel("y threshold (MeV)")
    axes[1].set_ylabel("RMS relative error")
    axes[1].set_title("Resolution stability")
    axes[1].legend()
    axes[2].plot(thresholds, n_vals, "o-", color="#10b981", linewidth=2)
    axes[2].axvline(x=10.0, color="#ef4444", linestyle=":", linewidth=1.5,
                    label="chosen = 10 MeV")
    
    axes[2].set_xlabel("y threshold (MeV)")
    axes[2].set_ylabel("N (events passing cut)")
    axes[2].set_title("Sample size")
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "threshold_stability_curves.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def energy_binned_comparison(method_preds, y_phys, save_dir,
                             bins=None):
    """Compare methods via Mean and RMS in energy bins.

    This produces a concrete per-energy-bin table and figure showing
    how each method performs across the full energy spectrum.
    Useful for diagnosing bias drift (e.g., WLS vs OLS).

    Parameters
    ----------
    method_preds : dict {method_name: y_hat_phys}
    y_phys : np.ndarray [N]   -- true physical energies
    save_dir : str
    bins : list of (lo, hi) tuples  -- energy bin edges in MeV
    """
    if bins is None:
        bins = [(10, 20), (20, 50), (50, 100), (100, 500), (500, 5000)]

    os.makedirs(save_dir, exist_ok=True)
    methods = list(method_preds.keys())
    table_data = {m: [] for m in methods}
    bin_labels = []
    bin_ns = []

    print("\n  Energy-binned comparison:")
    header = f"  {'Bin':>14s}  {'N':>6s}"

    for m in methods:
        header += f"  {m+' Mean':>14s}  {m+' RMS':>12s}"

    print(header)
    print("  " + "-" * len(header))

    for lo, hi in bins:
        mask = (y_phys >= lo) & (y_phys < hi)
        n = int(mask.sum())
        if n == 0:
            continue

        bin_labels.append(f"{lo}-{hi}")
        bin_ns.append(n)
        line = f"  {lo:5d}-{hi:4d} MeV  {n:6d}"

        for m in methods:
            r = (method_preds[m][mask] - y_phys[mask]) / y_phys[mask]
            mean_r = float(np.mean(r))
            rms_r = float(np.sqrt(np.mean(r ** 2)))
            table_data[m].append({"mean": mean_r, "rms": rms_r})
            line += f"  {mean_r:14.6f}  {rms_r:12.6f}"

        print(line)

    if len(bin_labels) == 0:
        return

    x = np.arange(len(bin_labels))
    n_methods = len(methods)
    width = 0.8 / n_methods
    colors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444"][:n_methods]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, m in enumerate(methods):
        means = [d["mean"] for d in table_data[m]]
        rms_vals = [d["rms"] for d in table_data[m]]

        offset = (i - n_methods / 2 + 0.5) * width
        axes[0].bar(x + offset, means, width, label=m, color=colors[i],
                    edgecolor="white", alpha=0.9)
        
        axes[1].bar(x + offset, rms_vals, width, label=m, color=colors[i],
                    edgecolor="white", alpha=0.9)

    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Mean relative error (bias)")
    axes[0].set_title("Bias per energy bin")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{bl}\nMeV" for bl in bin_labels], fontsize=8)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("RMS relative error")
    axes[1].set_title("Resolution per energy bin")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{bl}\nMeV" for bl in bin_labels], fontsize=8)
    axes[1].legend(fontsize=8)

    for j, n in enumerate(bin_ns):
        axes[0].text(j, axes[0].get_ylim()[0], f"N={n}", ha="center",
                     va="top", fontsize=7, color="gray")

    plt.tight_layout()
    path = os.path.join(save_dir, "energy_binned_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_relative_error_hist(r, stats, save_dir, label="", n_bins=200, filename="relative_error_hist.png", fit_gaussian=True):
    """REQUIRED PLOT 1: 1D histogram of (y_hat - y) / y.

    Parameters
    ----------
    r : np.ndarray
    stats : dict from compute_summary_stats
    save_dir : str
    label : str
        Descriptive label for title (e.g. "OLS (test)")
    n_bins : int
    filename : str
    fit_gaussian : bool  -- whether to overlay a Gaussian fit
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    # To make the plot readable despite potential extreme outliers,
    # clip the visualized range strictly (e.g., to the 99.5th percentile)
    r_abs = np.abs(r)
    if len(r_abs) > 0:
        clip_val = np.percentile(r_abs, 99.5)
        clip_val = max(clip_val, 1e-6)

    else:
        clip_val = 1.0

    mask = r_abs <= clip_val
    r_vis = r[mask]

    counts, bins, _ = ax.hist(
        r_vis, bins=n_bins, density=True, alpha=0.7, color="#3b82f6"
    )

    if fit_gaussian:
        x_norm = np.linspace(-clip_val, clip_val, 500)
        y_norm = norm.pdf(x_norm, loc=stats["mean"], scale=stats["std"])
        ax.plot(x_norm, y_norm, "r--", linewidth=2, alpha=0.8,
                label="Gaussian ref (matched mean/std)")

    ax.axvline(stats["mean"], color="#10b981", linewidth=1.5, linestyle="-",
               label=f"Mean = {stats['mean']:.4f}")

    stats_text = (
        f"Mean = {stats['mean']:.6f}\n"
        f"RMS  = {stats['rms']:.6f}\n"
        f"Std  = {stats['std']:.6f}\n"
        f"N    = {stats['n_samples']}"
    )

    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9,
                 edgecolor="#d1d5db")
    
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right", bbox=bbox_props,
            family="monospace")

    ax.set_xlabel(r"$(\ \hat{y} - y) / y$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    title_suffix = f" -- {label}" if label else ""
    ax.set_title(f"Relative error distribution{title_suffix}", fontsize=13)
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_relative_error_vs_energy(r, y, save_dir, label="", filename="relative_error_vs_energy.png"):
    """REQUIRED PLOT 2: 2D distribution of (y_hat - y) / y vs y.

    Parameters
    ----------
    r : np.ndarray  -- relative error
    y : np.ndarray  -- true energy (same length as r, already filtered)
    save_dir : str
    label : str
    filename : str
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    # 2D hexbin plot for better density visualization
    hb = ax.hexbin(y, r, gridsize=100, cmap='inferno', bins='log', mincnt=1, edgecolors='none')
    cb = fig.colorbar(hb, ax=ax, label="Log(Count)")
    ax.axhline(y=0, color="white", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("True energy y (MeV)", fontsize=12)
    ax.set_ylabel(r"$(\ \hat{y} - y) / y$", fontsize=12)
    title_suffix = f" -- {label}" if label else ""
    ax.set_title(f"Relative error vs. true energy{title_suffix}", fontsize=13)
    r_99 = np.percentile(np.abs(r), 99)
    ax.set_ylim(-r_99 * 1.2, r_99 * 1.2)
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_method_comparison(stats_dict, save_dir):
    """Optional: bar chart comparing mean/RMS across methods.

    Parameters
    ----------
    stats_dict : dict  {method_name: stats_dict}
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)
    methods = list(stats_dict.keys())
    means = [stats_dict[m]["mean"] for m in methods]
    rms_vals = [stats_dict[m]["rms"] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444"][:len(methods)]
    axes[0].bar(methods, means, color=colors, edgecolor="white", alpha=0.9)
    axes[0].set_ylabel("Mean relative error")
    axes[0].set_title("Bias comparison")
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    for i, v in enumerate(means):
        axes[0].text(i, v, f"{v:.5f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)

    axes[1].bar(methods, rms_vals, color=colors, edgecolor="white", alpha=0.9)
    axes[1].set_ylabel("RMS relative error")
    axes[1].set_title("Resolution comparison")

    for i, v in enumerate(rms_vals):
        axes[1].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "method_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

def save_metrics(all_metrics, path):
    """Save metrics dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"  Saved: {path}")


def print_metrics(stats, label=""):
    """Pretty-print a stats dict."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Mean = {stats['mean']:.6f}")
    print(f"{prefix}RMS  = {stats['rms']:.6f}")
    print(f"{prefix}Std  = {stats['std']:.6f}")
    print(f"{prefix}N    = {stats['n_samples']}")


def plot_time_estimate_hist(tau, save_dir, label="OF-time", min_energy_mask=None):
    """Histogram of estimated time offsets (in BC units).

    Parameters
    ----------
    tau : np.ndarray [N]  -- estimated time offsets
    save_dir : str
    label : str
    min_energy_mask : np.ndarray [N] bool or None
        If provided, only plot events passing this mask (signal events).
    """
    os.makedirs(save_dir, exist_ok=True)
    if min_energy_mask is not None:
        tau = tau[min_energy_mask]

    mean_t = float(np.mean(tau))
    std_t = float(np.std(tau))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tau, bins=100, color="#6366f1", edgecolor="white", alpha=0.85,
            range=(mean_t - 5 * std_t, mean_t + 5 * std_t))
    
    ax.axvline(mean_t, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_t:.4f} BC")
    
    ax.set_xlabel("Estimated time offset (BC units)")
    ax.set_ylabel("Count")
    ax.set_title(f"Bonus: Estimated signal timing [{label}]")
    ax.legend(loc="upper right", fontsize=10)
    ax.text(0.98, 0.85, f"Std = {std_t:.4f} BC\nN = {len(tau)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, "time_estimate_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

