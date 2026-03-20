import numpy as np

def compute_of_weights(g, C):
    """Derive full OF2 amplitude weights from template and noise covariance.

    Minimizes the variance w^T C w subject to three TileCal OF2 constraints:
        1. w^T g = 1        (unbiased amplitude)
        2. w^T g' = 0       (robust to small timing shifts)
        3. sum(w) = 0       (robust to constant pedestal offsets)

    Parameters
    ----------
    g : np.ndarray [N]     -- pulse template
    C : np.ndarray [N, N]  -- noise covariance matrix

    Returns
    -------
    w : np.ndarray [N]     -- OF2 amplitude weights
    """
    N = len(g)
    g_prime = np.zeros_like(g)
    g_prime[0] = g[1] - g[0]
    g_prime[-1] = g[-1] - g[-2]

    for i in range(1, N - 1):
        g_prime[i] = (g[i + 1] - g[i - 1]) / 2.0
        
    # The constraints are: V^T w = lambda_vec
    # where V = [g, g', 1] and lambda_vec = [1, 0, 0]
    ones = np.ones_like(g)
    V = np.column_stack([g, g_prime, ones])
    C_reg = C + 1e-10 * np.eye(N)
    
    # Solve C * X = V instead of explicitly inverting C for numerical stability
    C_inv_V = np.linalg.solve(C_reg, V)
    
    # Solve the Lagrange system: Q * lambda = targets
    # Q = V^T C^{-1} V is a 3x3 matrix
    Q = V.T @ C_inv_V
    
    # Target values for constraints [amplitude=1, time_drift=0, pedestal_shift=0]
    targets = np.array([1.0, 0.0, 0.0])
    
    # Solve for Lagrange multipliers lambda = Q^{-1} targets
    lambdas = np.linalg.solve(Q, targets)
    
    # Compute final weights: w = C^{-1} V lambda
    w = C_inv_V @ lambdas
    
    return w


def predict_of(X, w, pedestal=None):
    """Apply OF weights to compute raw amplitude estimates.

    Parameters
    ----------
    X : np.ndarray [N, 7]
    w : np.ndarray [7]
    pedestal : np.ndarray [7] or None
        If provided, subtract pedestal before applying weights.

    Returns
    -------
    y_hat_raw : np.ndarray [N]
    """
    if pedestal is not None:
        X_centered = X - pedestal[np.newaxis, :]

    else:
        X_centered = X

    y_hat_raw = X_centered @ w

    return y_hat_raw


def calibrate_of(y_hat_raw, y_true):
    """Fit a linear calibration y = alpha * y_hat_raw + beta.

    This corrects any residual scale/offset between the OF amplitude
    estimate and the actual target energy, using least-squares.

    Parameters
    ----------
    y_hat_raw : np.ndarray [N]
    y_true : np.ndarray [N]

    Returns
    -------
    alpha : float
    beta : float
    """
    # Linear fit: y_true = alpha * y_hat_raw + beta
    A = np.stack([y_hat_raw, np.ones_like(y_hat_raw)], axis=1)
    result, _, _, _ = np.linalg.lstsq(A, y_true, rcond=None)
    alpha, beta = result[0], result[1]

    return alpha, beta


def predict_of_calibrated(X, w, alpha, beta, pedestal=None):
    """Full OF prediction with calibration.

    Parameters
    ----------
    X : np.ndarray [N, 7]
    w : np.ndarray [7]
    alpha : float
    beta : float
    pedestal : np.ndarray [7] or None

    Returns
    -------
    y_hat : np.ndarray [N]
    """
    y_raw = predict_of(X, w, pedestal)

    return alpha * y_raw + beta


def compute_of_time_weights(g, C):
    """Derive full OF2 time weights from the template derivative and noise covariance.

    Minimizes w_t^T C w_t subject to three TileCal OF2 constraints:
        1. w_t^T g  = 0   (zero response to pure amplitude)
        2. w_t^T g' = 1   (unbiased response to timing shifts)
        3. sum(w_t) = 0   (robust to constant pedestal offsets)

    Parameters
    ----------
    g : np.ndarray [7]     -- pulse template (normalized, peak=1)
    C : np.ndarray [7, 7]  -- noise covariance matrix

    Returns
    -------
    w_t : np.ndarray [7]   -- OF2 time weights
    g_prime : np.ndarray [7] -- the derivative template used
    """
    N = len(g)
    g_prime = np.zeros_like(g)
    g_prime[0] = g[1] - g[0]
    g_prime[-1] = g[-1] - g[-2]

    for i in range(1, N - 1):
        g_prime[i] = (g[i + 1] - g[i - 1]) / 2.0

    ones = np.ones_like(g)
    V = np.column_stack([g, g_prime, ones])
    C_reg = C + 1e-10 * np.eye(N)
    C_inv_V = np.linalg.solve(C_reg, V)
    
    # Q = V^T C^{-1} V is a 3x3 matrix
    Q = V.T @ C_inv_V

    # Target values for constraints [amplitude=0, time_drift=1, pedestal_shift=0]
    targets = np.array([0.0, 1.0, 0.0])

    try:
        lambdas = np.linalg.solve(Q, targets)
        w_t = C_inv_V @ lambdas

    except np.linalg.LinAlgError:
        u = np.linalg.solve(C_reg, g_prime)
        norm = g_prime @ u
        w_t = u / norm if norm > 1e-15 else np.zeros_like(g)

    return w_t, g_prime


def predict_of_time(X, w_t, pedestal=None):
    """Apply OF time weights to estimate the signal phase.

    Parameters
    ----------
    X : np.ndarray [N, 7]
    w_t : np.ndarray [7]
    pedestal : np.ndarray [7] or None

    Returns
    -------
    tau_hat : np.ndarray [N]  -- estimated time offset in BC units
    """
    if pedestal is not None:
        X_centered = X - pedestal[np.newaxis, :]

    else:
        X_centered = X
        
    return X_centered @ w_t
