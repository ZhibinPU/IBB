"""Synthetic covariance structures for the simulation study.

Three population correlation matrices, each a standard stress test for a
covariance estimator in the d > n regime:

* :func:`block_matrix` — equicorrelated blocks (off-diagonal 0.6 within a
  block, 0 across). Tests whether an estimator recovers sharp group structure.
* :func:`toeplitz_matrix` — ``0.75^|i-j|``, decaying but never exactly zero.
  Tests behaviour when the truth is dense-but-weak, which shrinkage can wash out.
* :func:`banded_matrix` — linear taper ``1 - |i-j|/10`` inside a bandwidth of
  10, exactly zero outside. Tests exact-sparsity recovery.

Each ``simulation_*`` draws ``n`` Gaussian samples with zero mean and returns
both the data and the population matrix that generated it, so estimation error
can be measured against a known target.

Note that the population matrices are *correlation* matrices (unit diagonal),
so ``Sigma`` is simultaneously the population covariance and correlation.
"""

import numpy as np


def banded_matrix(dimension, bandwidth=10):
    """Banded correlation matrix: ``1 - |i-j|/bandwidth`` inside the band, else 0."""
    x_true = np.eye(dimension)
    for i in range(dimension):
        for j in range(dimension):
            if abs(i - j) <= bandwidth:
                x_true[i][j] = 1 - abs(i - j) / bandwidth
    return x_true


def block_matrix(dimension, group):
    """Block-equicorrelated matrix: ``group`` equal blocks, 0.6 within, 0 across.

    ``dimension`` should be divisible by ``group``; any remainder is left as
    identity rows at the end (the original behaviour, since the block loop
    covers only ``group * (dimension // group)`` rows).
    """
    x_true = np.eye(dimension)
    part = int(dimension / group)

    x_submatrix = np.eye(part)
    for i in range(part):
        for j in range(part):
            if i != j:
                x_submatrix[i][j] = 0.6

    for k in range(0, group):
        x_true[k * part:(k + 1) * part, k * part:(k + 1) * part] = x_submatrix

    return x_true


def toeplitz_matrix(dimension, rho=0.75):
    """Toeplitz correlation matrix with entries ``rho^|i-j|``."""
    x_true = np.eye(dimension)
    for i in range(dimension):
        for j in range(dimension):
            x_true[i][j] = rho ** abs(i - j)
    return x_true


def _draw(Sigma, n, random_state):
    """Draw ``n`` zero-mean Gaussian samples with covariance ``Sigma``."""
    d = Sigma.shape[0]
    np.random.seed(random_state)
    return np.random.multivariate_normal(np.zeros(d), Sigma, size=n)


def simulation_block(n=1000, d=5, random_state=None, group=5):
    """Gaussian data from a block-equicorrelated population covariance.

    Returns
    -------
    samples : ndarray, shape (n, d)
    Sigma : ndarray, shape (d, d)
        The population covariance used to draw ``samples``.
    """
    Sigma = block_matrix(d, group)
    return _draw(Sigma, n, random_state), Sigma


def simulation_band(n=1000, d=5, random_state=None, bandwidth=10):
    """Gaussian data from a banded population covariance.

    Returns
    -------
    samples : ndarray, shape (n, d)
    Sigma : ndarray, shape (d, d)
    """
    Sigma = banded_matrix(d, bandwidth)
    return _draw(Sigma, n, random_state), Sigma


def simulation_toeplitz(n=1000, d=5, random_state=None, rho=0.75):
    """Gaussian data from a Toeplitz population covariance.

    Returns
    -------
    samples : ndarray, shape (n, d)
    Sigma : ndarray, shape (d, d)
    """
    Sigma = toeplitz_matrix(d, rho)
    return _draw(Sigma, n, random_state), Sigma
