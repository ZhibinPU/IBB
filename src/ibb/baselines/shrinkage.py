"""Classical shrinkage and factor-analytic covariance baselines.

Three estimators, all returning a ``(d, d)`` covariance with a small positive
diagonal bias added for conditioning:

* :func:`cov_ledoitwolf` — Ledoit-Wolf shrinkage towards a scaled identity
* :func:`cov_oas` — oracle-approximating shrinkage, same target, different
  shrinkage intensity
* :func:`cov_ppca` — probabilistic PCA, i.e. a ``k``-factor model
  ``W Wᵀ + sigma² I``

:func:`cov_ppca` is load-bearing beyond the baseline comparison: it also
supplies the population covariance for the synthetic data in
:mod:`ibb.synthetic`.
"""

import numpy as np
from sklearn.covariance import OAS, LedoitWolf
from sklearn.decomposition import PCA


def _diag_bias_value(S, diag_bias=1e-6, mode="relative"):
    """Resolve a diagonal-bias magnitude.

    Parameters
    ----------
    S : ndarray, shape (d, d)
    diag_bias : float or None
        ``None`` means no bias.
    mode : {"relative", "absolute"}
        ``"relative"`` scales by ``mean(diag(S))``, so the bias tracks the data
        scale; ``"absolute"`` uses ``diag_bias`` as-is.
    """
    if diag_bias is None:
        return 0.0

    if mode == "absolute":
        return float(diag_bias)

    if mode == "relative":
        scale = float(np.mean(np.diag(S)))
        # Guard against a degenerate or non-finite scale.
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return float(diag_bias) * scale

    raise ValueError("mode must be 'relative' or 'absolute'")


def add_diagonal_bias(S, diag_bias=1e-6, mode="relative"):
    """Add a positive multiple of the identity to improve conditioning.

    With a positive bias the result is SPD as long as ``S`` is at least PSD,
    which matters downstream: the QDA in :mod:`ibb.evaluation` has to invert it.
    """
    S = np.asarray(S, dtype=float)
    b = _diag_bias_value(S, diag_bias=diag_bias, mode=mode)
    return S + b * np.eye(S.shape[0])


def cov_oas(X, assume_centered=False, diag_bias=1e-6, bias_mode="relative"):
    """Oracle-approximating-shrinkage covariance of ``X`` (n, d) -> (d, d)."""
    X = np.asarray(X, dtype=float)
    est = OAS(assume_centered=assume_centered).fit(X)
    return add_diagonal_bias(est.covariance_, diag_bias=diag_bias, mode=bias_mode)


def cov_ledoitwolf(X, assume_centered=False, diag_bias=1e-6, bias_mode="relative"):
    """Ledoit-Wolf shrinkage covariance of ``X`` (n, d) -> (d, d)."""
    X = np.asarray(X, dtype=float)
    est = LedoitWolf(assume_centered=assume_centered).fit(X)
    return add_diagonal_bias(est.covariance_, diag_bias=diag_bias, mode=bias_mode)


def cov_ppca(X, n_components, diag_bias=1e-6, bias_mode="relative"):
    """PPCA covariance: ``Sigma = U diag(lambda_k - sigma²) Uᵀ + sigma² I``.

    PPCA is approximated in closed form from the sample eigen-decomposition
    (the standard practical route, avoiding EM): the top ``k`` eigenpairs form
    the latent part, and ``sigma²`` is the mean of the discarded eigenvalues.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    n_components : int
        Latent dimension ``k``; must satisfy ``1 <= k < d``.
    diag_bias, bias_mode
        See :func:`add_diagonal_bias`.

    Returns
    -------
    ndarray, shape (d, d)

    Notes
    -----
    When ``n <= k`` there are no discarded eigenvalues to average, so
    ``sigma²`` falls back to a floor of ``1e-8 * mean(eigenvalues)``. The latent
    variances are clamped at zero, keeping the result PSD.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    k = int(n_components)
    if not (1 <= k < d):
        raise ValueError(f"n_components must be in [1, d-1], got {k} with d={d}")

    # PPCA assumes centered observations.
    Xc = X - X.mean(axis=0, keepdims=True)

    pca = PCA(n_components=min(n, d), svd_solver="full")
    pca.fit(Xc)

    evals = np.asarray(pca.explained_variance_, dtype=float)  # descending
    evecs = np.asarray(pca.components_, dtype=float)  # (m, d), rows are eigenvectors

    lam_top = evals[:k]
    U_top = evecs[:k].T  # (d, k)

    # Isotropic noise variance from the discarded spectrum.
    sigma2 = float(np.mean(evals[k:])) if len(evals) > k else 0.0

    avg_var = float(np.mean(evals)) if len(evals) > 0 else 1.0
    sigma2 = max(sigma2, 1e-8 * avg_var)

    # Latent part accounts only for variance above the noise floor.
    latent = np.maximum(lam_top - sigma2, 0.0)
    S = (U_top * latent) @ U_top.T + sigma2 * np.eye(d)

    return add_diagonal_bias(S, diag_bias=diag_bias, mode=bias_mode)
