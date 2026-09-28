"""Bootstrap-aggregated ("bagging") correlation estimator.

The comparison method used throughout the paper: resample rows with
replacement, compute a standardised cross-product ("kinship") matrix on each
resample, and average.  Averaging over resamples regularises the estimate in
the d > n regime much the way shrinkage does, but without a prior.
"""

import numpy as np


def calculate_kinship(W, center=False):
    """Standardised cross-product matrix of ``W`` (n, m) -> (m, m).

    Parameters
    ----------
    W : ndarray, shape (n, m)
        Data matrix; columns are standardised in place (unbiased sd, ddof=1).
    center : bool
        If True, rescale by the trace of the doubly-centred matrix so that
        ``trace(P K P) == n - 1`` — the normalisation used in kinship /
        heritability work.
    """
    n = W.shape[0]
    W = (W - np.mean(W, axis=0)) / np.std(W, axis=0, ddof=1)

    K = W.T @ W * 1.0 / (float(n) - 1)

    if center:
        P = np.diag(np.repeat(1, n)) - 1 / float(n) * np.ones((n, n))
        S = np.trace((P @ K) @ P)
        return (n - 1) * K / S

    return K


def bagging(Y, sample_size=10, iterations=100):
    """Average :func:`calculate_kinship` over bootstrap resamples of ``Y``.

    The seed is set to the loop index on each iteration, so the result is
    reproducible for a given ``(sample_size, iterations)`` — this matches the
    original implementation and is what the reported numbers were produced with.
    """
    R_samples = np.zeros((Y.shape[1], Y.shape[1]))

    for i in range(iterations):
        np.random.seed(i)
        idx = np.random.choice(Y.shape[0], size=sample_size, replace=True)
        sample = Y[idx, :]
        R_samples = R_samples + calculate_kinship(sample)

    return R_samples / iterations


def fit_cov_bag(X, sample_size=None, iterations=None):
    """Bagged *covariance* estimate: bagged correlation rescaled by sample sds.

    Defaults follow the experiments: ``sample_size = n`` and ``iterations = d``.
    """
    X = np.asarray(X)
    n, d = X.shape
    if sample_size is None:
        sample_size = n
    if iterations is None:
        iterations = d

    R_hat = bagging(X, sample_size=sample_size, iterations=iterations)
    sd = np.sqrt(np.diag(np.cov(X.T, ddof=1)))
    return np.diag(sd) @ R_hat @ np.diag(sd)
