"""POET: Principal Orthogonal complEment Thresholding.

Fan, Liao & Mincheva (2013), *JASA*. The estimator splits the sample covariance
into a low-rank factor part (the top ``K`` eigenpairs) and an idiosyncratic
remainder, then adaptively soft- or hard-thresholds the remainder to make it
sparse:

    Sigma_hat = Sigma_f + threshold(S - Sigma_f)

The threshold ``c * sqrt(log(p)/n)``, scaled per entry by the idiosyncratic
standard deviations, is the rate from the paper. The diagonal of the
idiosyncratic part is restored after thresholding — shrinking marginal variances
towards zero would bias the whole estimate.
"""

import numpy as np
from numpy.linalg import eigh


def soft_threshold(matrix, tau):
    """Elementwise soft-thresholding: ``sign(x) * max(|x| - tau, 0)``.

    ``tau`` may be a scalar or an array broadcastable against ``matrix``.
    """
    return np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0.0)


def poet(X, K=None, threshold_method="soft", c=0.5):
    """POET covariance estimate of ``X``.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Rows are observations, columns are variables.
    K : int, optional
        Number of factors. If ``None``, chosen by the eigenvalue-ratio rule over
        the first ten ratios.
    threshold_method : {"soft", "hard"}
        Thresholding rule for the idiosyncratic covariance.
    c : float
        Scaling constant on the threshold.

    Returns
    -------
    Sigma_hat : ndarray, shape (p, p)
        ``Sigma_f + Sigma_u`` (thresholded).
    Sigma_f : ndarray, shape (p, p)
        Low-rank factor component.
    Sigma_u : ndarray, shape (p, p)
        Thresholded idiosyncratic component.
    """
    n, p = X.shape
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Sample covariance (1/n, matching the paper rather than the unbiased 1/(n-1))
    S = (X_centered.T @ X_centered) / n

    # Eigen-decomposition, sorted descending.
    eigvals, eigvecs = eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Number of factors by the eigenvalue-ratio rule, if not supplied.
    if K is None:
        ratios = eigvals[:-1] / eigvals[1:]
        K = np.argmax(ratios[:min(10, len(ratios))]) + 1

    # Factor component from the top-K eigenpairs.
    # (The original wrote sqrt(diag(l)) @ sqrt(diag(l)), which is just diag(l);
    # taking the square root first would also produce NaNs if a leading
    # eigenvalue came out marginally negative.)
    V = eigvecs[:, :K]
    Sigma_f = V @ np.diag(eigvals[:K]) @ V.T

    # Idiosyncratic remainder.
    Sigma_u = S - Sigma_f

    # Adaptive threshold: tau_ij = c * sqrt(log(p)/n) * sd_i * sd_j
    sigma_diag = np.sqrt(np.diag(Sigma_u))
    tau = c * np.sqrt(np.log(p) / n)
    threshold_matrix = tau * np.outer(sigma_diag, sigma_diag)

    if threshold_method == "soft":
        Sigma_u_thresh = soft_threshold(Sigma_u, threshold_matrix)
    elif threshold_method == "hard":
        Sigma_u_thresh = Sigma_u * (np.abs(Sigma_u) > threshold_matrix)
    else:
        raise ValueError("threshold_method must be 'soft' or 'hard'")

    # Keep marginal variances intact.
    np.fill_diagonal(Sigma_u_thresh, np.diag(Sigma_u))

    return Sigma_f + Sigma_u_thresh, Sigma_f, Sigma_u_thresh


def fit_cov_pot(X, K=10, threshold_method="soft", c=0.5):
    """Return only ``Sigma_hat``, to match the other ``fit_cov_*`` baselines.

    ``K=10`` is the value used in the experiments.
    """
    Sigma_hat, _Sigma_f, _Sigma_u = poet(
        X, K=K, threshold_method=threshold_method, c=c
    )
    return Sigma_hat
