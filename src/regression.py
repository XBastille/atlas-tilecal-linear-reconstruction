import os
import numpy as np
import matplotlib.pyplot as plt
from src.io import denormalize

def _augment(X):
    """Append a column of ones for the bias term."""
    return np.column_stack([X, np.ones(X.shape[0])])


def fit_ridge(X, y, lambda_reg):
    """Solve ridge regression for a single regularization strength.

    Parameters
    ----------
    X : np.ndarray [N, 7]
    y : np.ndarray [N]
    lambda_reg : float

    Returns
    -------
    w : np.ndarray [7]
    b : float
    """
    X_aug = _augment(X)  # [N, 8]
    n_features = X_aug.shape[1]
    Lambda = lambda_reg * np.eye(n_features)
    Lambda[-1, -1] = 0.0  

    # Normal equation: (X^T X + Lambda)^{-1} X^T y
    A = X_aug.T @ X_aug + Lambda
    rhs = X_aug.T @ y
    theta = np.linalg.solve(A, rhs)
    w = theta[:-1]
    b = theta[-1]

    return w, b


def fit_wls_ridge(X, y, y_phys, lambda_reg, min_energy=10.0, wls_cap=None,
                  wls_power=2.0, wls_balance_bins=None):
    """Weighted Least Squares ridge regression.

    Minimizes sum_i  (y_hat_i - y_i)^2 * w_i  +  lambda * ||w||^2
    where w_i combines:
      - a relative-error style term: 1 / max(y_phys_i, wls_cap)^wls_power
      - an optional bin-balancing term: 1 / N_bin(i)

    This directly optimizes the sum of squared relative errors (if power=2),
    aligning the training objective with the evaluation metric
    r = (y_hat - y) / y.  Only events with y_phys > min_energy are
    used to avoid unstable weights on noise-only BCs.

    Parameters
    ----------
    X : np.ndarray [N, 7]       -- samples (normalized)
    y : np.ndarray [N]          -- targets (normalized)
    y_phys : np.ndarray [N]     -- targets (physical/denormalized MeV)
    lambda_reg : float
    min_energy : float          -- minimum physical energy for inclusion
    wls_cap : float or None     -- if set, cap the weight denominator from below
                                   to prevent low-energy bins from dominating
    wls_power : float           -- exponent for the denominator (default 2.0)
    wls_balance_bins : list or None -- if set, bin edges for balancing weights.
                                       Weights are multiplied by 1/N_bin.

    Returns
    -------
    w : np.ndarray [7]
    b : float
    """
    mask = y_phys > min_energy
    X_sel = X[mask]
    y_sel = y[mask]
    y_phys_sel = y_phys[mask]
    X_aug = _augment(X_sel)  # [M, 8]
    n_features = X_aug.shape[1]
    if wls_cap is not None:
        w_den = np.maximum(y_phys_sel, wls_cap)

    else:
        w_den = y_phys_sel

    W = 1.0 / (w_den ** wls_power)
    if wls_balance_bins is not None:
        # Digitize returns indices 1 to len(bins)
        inds = np.digitize(y_phys_sel, wls_balance_bins)
        counts = np.bincount(inds)
        counts[counts == 0] = 1
        # Multiply by 1 / counts[bin_index]
        W = W / counts[inds]
        W = W * (len(W) / np.sum(W))

    # WLS normal equation: (X^T W X + lambda*I) theta = X^T W y
    # W is diagonal, so X^T W X = sum_i w_i * x_i x_i^T
    XtWX = X_aug.T @ (W[:, None] * X_aug)
    XtWy = X_aug.T @ (W * y_sel)
    reg = lambda_reg * np.eye(n_features)
    reg[-1, -1] = 0.0  
    theta = np.linalg.solve(XtWX + reg, XtWy)
    w = theta[:-1]
    b = theta[-1]
    
    return w, b


def predict_ridge(X, w, b):
    """Apply ridge regression weights.

    Parameters
    ----------
    X : np.ndarray [N, 7]
    w : np.ndarray [7]
    b : float

    Returns
    -------
    y_hat : np.ndarray [N]
    """
    return X @ w + b


def evaluate_lambda_grid(X_train, y_train, y_train_phys, X_val, y_val_phys,
                         lambda_grid, y_stats, min_energy=10.0, wls_kwargs=None):
    """Grid search over regularizations for Ridge or WLS Ridge.

    Trains on (X_train, y_train) and evaluates on the validation set using
    the RMS of relative error (computed with physical units and threshold).

    Parameters
    ----------
    X_train : np.ndarray [N, 7]
    y_train : np.ndarray [N]
    y_train_phys : np.ndarray [N] (used for WLS)
    X_val : np.ndarray [N, 7]
    y_val_phys : np.ndarray [N]
    lambda_grid : list of float
    y_stats : dict
    min_energy : float
    wls_kwargs : dict or None  -- if set, uses fit_wls_ridge

    Returns
    -------
    best_w : np.ndarray [7]
    best_b : float
    best_lambda : float
    scan_results : list of dict
    """
    results = []
    best_rms = np.inf
    best_w, best_b, best_lambda = None, None, None

    for lam in lambda_grid:
        if wls_kwargs is not None:
            w, b = fit_wls_ridge(X_train, y_train, y_train_phys, lam,
                                 min_energy=min_energy, **wls_kwargs)
            
        else:
            w, b = fit_ridge(X_train, y_train, lam)

        y_hat_val = predict_ridge(X_val, w, b)
        y_hat_val_phys = denormalize(y_hat_val, y_stats["mean_lo"], y_stats["std_lo"])
        mask = y_val_phys > min_energy
        if mask.sum() > 0:
            r = (y_hat_val_phys[mask] - y_val_phys[mask]) / y_val_phys[mask]
            rms = np.sqrt(np.mean(r ** 2))

        else:
            rms = np.inf

        results.append({
            "lambda": lam,
            "rms_relative": rms,
            "w": w.copy(),
            "b": b
        })

        if rms < best_rms:
            best_rms = rms
            best_w = w.copy()
            best_b = b
            best_lambda = lam

    return best_w, best_b, best_lambda, results


def compare_weights(w_of, w_ridge, save_dir):
    """Side-by-side bar plot of OF weights vs. ridge regression weights.

    Parameters
    ----------
    w_of : np.ndarray [7]
    w_ridge : np.ndarray [7]
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(w_of)
    positions = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_of = ax.bar(positions - width / 2, w_of, width,
                     label="OF-style", color="#6366f1", edgecolor="white", alpha=0.9)
    
    bars_rr = ax.bar(positions + width / 2, w_ridge, width,
                     label="Ridge regression", color="#f59e0b", edgecolor="white", alpha=0.9)

    ax.set_xlabel("Sample position in window")
    ax.set_ylabel("Weight value")
    ax.set_title("Linear weights comparison: OF-style vs. Ridge regression")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"s{i}" for i in positions])
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    for bar, val in zip(bars_of, w_of):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * np.sign(bar.get_height()),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=7, color="#6366f1")

    for bar, val in zip(bars_rr, w_ridge):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * np.sign(bar.get_height()),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=7, color="#f59e0b")

    plt.tight_layout()
    path = os.path.join(save_dir, "weights_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_lambda_scan(results, save_dir):
    """Plot RMS relative error vs. lambda from the grid search."""
    os.makedirs(save_dir, exist_ok=True)
    lambdas = [r["lambda"] for r in results]
    rms_vals = [r["rms_relative"] for r in results]

    # Handle lambda=0 (OLS) gracefully for semilogx
    if 0.0 in lambdas:
        zero_idx = lambdas.index(0.0)
        ols_rms = rms_vals.pop(zero_idx)
        lambdas.pop(zero_idx)

    else:
        ols_rms = None

    fig, ax = plt.subplots(figsize=(8, 5))
    
    if lambdas:
        ax.semilogx(lambdas, rms_vals, "o-", color="#ef4444", markersize=6, label="Ridge")
        
    if ols_rms is not None:
        ax.axhline(ols_rms, color="#3b82f6", linestyle="--", linewidth=1.5, 
                   label=f"OLS baseline ($\lambda=0$)")

    ax.set_xlabel("Regularization strength (lambda)")
    ax.set_ylabel("RMS relative error (validation)")
    ax.set_title("Ridge regression: lambda scan")
    ax.grid(True, alpha=0.3)
    best_rms = min(r["rms_relative"] for r in results)
    best_lam = next(r["lambda"] for r in results if r["rms_relative"] == best_rms)
    if best_lam > 0.0:
        ax.axvline(best_lam, color="#10b981", linestyle="--", alpha=0.7,
                   label=f"Best pos. lambda={best_lam:.1e}")

    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "lambda_scan.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
