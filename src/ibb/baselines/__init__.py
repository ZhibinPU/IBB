"""Comparison covariance estimators.

All are ``fit_cov_*(X) -> (d, d)`` or close to it, so they can be swapped for
:func:`ibb.fit_cov_ibb` in the benchmarks and the CV harness.
"""

from .bagging import bagging, calculate_kinship, fit_cov_bag
from .poet import fit_cov_pot, poet, soft_threshold
from .shrinkage import (
    add_diagonal_bias,
    cov_ledoitwolf,
    cov_oas,
    cov_ppca,
)

__all__ = [
    # bagging
    "bagging",
    "calculate_kinship",
    "fit_cov_bag",
    # POET
    "poet",
    "fit_cov_pot",
    "soft_threshold",
    # classical shrinkage / factor models
    "cov_ppca",
    "cov_ledoitwolf",
    "cov_oas",
    "add_diagonal_bias",
]
