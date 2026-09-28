"""Density evaluation, multiple-testing correction, and small helpers."""

import numpy as np
from numpy.linalg import eigh, slogdet, solve


def cov_to_corr(S):
    """Covariance -> correlation.  NaNs (zero-variance rows) are set to 0."""
    S = np.asarray(S)
    sd = np.sqrt(np.diag(S))
    R = S / np.outer(sd, sd)
    R[np.isnan(R)] = 0
    return R


def nearest_spd(A, floor=1e-10):
    """Project a symmetric matrix onto the SPD cone by clipping eigenvalues."""
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    w, V = eigh(A)
    w = np.maximum(w, floor)
    A_spd = (V * w) @ V.T
    return 0.5 * (A_spd + A_spd.T)


def logpdf_mvn(x, mu, Sigma, jitter=1e-6):
    """log N(x | mu, Sigma), robust to a marginally indefinite ``Sigma``.

    Shrinkage estimates in the d > n regime often land just outside the SPD
    cone numerically.  A jittered Cholesky-free path is tried first; if the
    determinant is still non-positive the matrix is projected onto the SPD cone
    via :func:`nearest_spd` rather than raising.
    """
    d = x.shape[0]
    Sigma = np.asarray(Sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)

    Sigma_j = Sigma + jitter * np.eye(d)
    sign, logdet = slogdet(Sigma_j)

    if sign <= 0:
        Sigma_j = nearest_spd(Sigma)
        sign, logdet = slogdet(Sigma_j)

    diff = x - mu
    quad = diff @ solve(Sigma_j, diff)

    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def logit_to_prob(logit):
    """Numerically plain sigmoid."""
    return 1.0 / (1.0 + np.exp(-logit))


def ci95_from_samples(samples):
    """Mean and empirical 95% interval (2.5% / 97.5% quantiles).

    Used on per-fold cross-validation metrics, where the fold-to-fold spread —
    not a parametric standard error — is the quantity of interest.
    """
    samples = np.asarray(samples, dtype=float)
    return samples.mean(), tuple(np.quantile(samples, [0.025, 0.975]))


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up correction.

    Returns
    -------
    qvals : ndarray
        BH-adjusted p-values, in the input order.
    reject : ndarray of bool
        ``qvals < alpha``.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)

    order = np.argsort(pvals)
    ranked_p = pvals[order]

    q = ranked_p * m / np.arange(1, m + 1)

    # Enforce monotonicity right-to-left, then truncate at 1.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    qvals = np.empty_like(q)
    qvals[order] = q

    return qvals, qvals < alpha


def bh_fdr_matrix(pmat, alpha=0.05):
    """Apply :func:`bh_fdr` to the upper triangle of a symmetric p-value matrix.

    Only the upper triangle (including the diagonal) enters the correction, so
    each network block is counted once; the result is mirrored back.
    """
    pmat = np.asarray(pmat, dtype=float)
    K = pmat.shape[0]
    iu = np.triu_indices(K)

    qvec, reject = bh_fdr(pmat[iu], alpha=alpha)

    qmat = np.full_like(pmat, np.nan, dtype=float)
    sigmat = np.zeros_like(pmat, dtype=bool)

    qmat[iu] = qvec
    sigmat[iu] = reject

    qmat[(iu[1], iu[0])] = qvec
    sigmat[(iu[1], iu[0])] = reject

    return qmat, sigmat


def frobenius(A, B):
    """Frobenius-norm distance ``||A - B||_F``."""
    return np.linalg.norm(np.asarray(A) - np.asarray(B), ord="fro")


def spectral(A, B):
    """Spectral-norm distance ``||A - B||_2``."""
    return np.linalg.norm(np.asarray(A) - np.asarray(B), ord=2)
