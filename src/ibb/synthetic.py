"""Synthetic data drawn from a covariance fitted to real data.

These generators give a *known* population covariance while keeping the
correlation structure of a real dataset, which is what makes an estimation-error
comparison meaningful: the target is realistic but still known exactly.

:func:`nonG_sim` draws from a multivariate t with 10 degrees of freedom instead
of a Gaussian, so the estimators are tested off their model assumption — the
Gaussian likelihood inside IBB is then misspecified by construction.

Both depend on ``cov_ppca``, which is still pending (see
:mod:`ibb.baselines.shrinkage`).
"""

import numpy as np
from scipy.stats import multivariate_t

from .baselines.shrinkage import cov_ppca

#: PPCA settings used for the synthetic-data covariance throughout.
PPCA_KWARGS = dict(n_components=12, diag_bias=1e-2, bias_mode="absolute")

#: Degrees of freedom of the heavy-tailed generator.
T_DF = 10


def G_sim(arr, n=100, random_state=None):
    """Gaussian draws from a PPCA covariance fitted to ``arr``.

    Returns
    -------
    samples : ndarray, shape (n, d)
    cov : ndarray, shape (d, d)
        The population covariance the samples were drawn from.
    """
    mean_ = np.mean(arr, axis=0)
    cov = cov_ppca(arr, **PPCA_KWARGS)

    np.random.seed(random_state)
    samples = np.random.multivariate_normal(mean_, cov, size=n)

    return samples, cov


def nonG_sim(arr, n=100, random_state=None, df=T_DF):
    """Multivariate-t draws from a PPCA shape matrix fitted to ``arr``.

    Note that ``cov`` is the *shape* matrix passed to the t distribution, not
    the covariance of the draws: for ``df`` degrees of freedom the true
    covariance is ``cov * df / (df - 2)``.  The error comparisons in the paper
    use ``cov`` as the target, so estimators are scored against the shape
    matrix — consistent across methods, but not the sampling covariance.
    """
    mean_ = np.mean(arr, axis=0)
    cov = cov_ppca(arr, **PPCA_KWARGS)

    samples = multivariate_t.rvs(
        loc=mean_, shape=cov, df=df, size=n, random_state=random_state
    )

    return samples, cov
