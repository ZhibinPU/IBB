"""High-level entry points: data in, covariance / shrinkage map out.

This module also collects the HMC step sizes and chain lengths that were tuned
per experiment.  In the original notebooks each experiment redefined its own
``fit_cov_ibb`` with different constants inlined, which made it easy to run one
experiment with another one's settings; :data:`HMC_PRESETS` keeps them in one
place and named.

The model is stiff and the step size does not transfer across (n, d) regimes —
if you move to a new data size, re-tune rather than reusing a preset, and watch
the acceptance rate printed by :func:`~ibb.hmc.hmc_sample`.
"""

import numpy as np
import torch

from .hmc import hmc_sample
from .model import CorrShrinkCovModel
from .posterior import reconstruct_corr, reconstruct_sigma, significant_mask

#: Chain settings used by each experiment, keyed by a short name.
#: ``num_steps`` is 10 and ``thin`` is 1 throughout.
HMC_PRESETS = {
    # --- simulation study: block covariance, d = 15n ---
    "sim_block_n200": dict(num_samples=330, burnin=330, step_size=0.000045),
    "sim_block_n300": dict(num_samples=400, burnin=400, step_size=0.000025),
    "sim_block_n400": dict(num_samples=200, burnin=1000, step_size=0.000017),
    # --- simulation study: Toeplitz covariance, d = 15n ---
    "sim_toeplitz": dict(num_samples=350, burnin=350, step_size=0.000025),
    # --- cortical-thickness application (rawCT_CG / rawCT_GG) ---
    "ct_real": dict(num_samples=50, burnin=50, step_size=0.002),
    "ct_synthetic": dict(num_samples=150, burnin=150, step_size=0.000015),
    # --- ADHD-200 application, 954 ROIs ---
    "adhd_cov": dict(num_samples=350, burnin=350, step_size=0.00022),
    "adhd_shrinkage": dict(num_samples=350, burnin=350, step_size=0.0003),
}

_DEFAULTS = dict(num_steps=10, thin=1)


def _resolve(preset, overrides):
    """Merge a preset name (or dict) with explicit keyword overrides."""
    if isinstance(preset, str):
        if preset not in HMC_PRESETS:
            raise KeyError(
                f"unknown preset {preset!r}; available: {sorted(HMC_PRESETS)}"
            )
        cfg = dict(HMC_PRESETS[preset])
    elif preset is None:
        cfg = {}
    else:
        cfg = dict(preset)
    return {**_DEFAULTS, **cfg, **overrides}


def _as_tensor(X, dtype=torch.float64):
    if isinstance(X, torch.Tensor):
        return X.to(dtype=dtype)
    return torch.tensor(np.asarray(X), dtype=dtype)


def run_chain(X, preset="adhd_cov", delta=1e-4, dtype=torch.float64, **hmc_kwargs):
    """Fit the model to ``X`` and return ``(samples, X_tensor)``.

    Use this when you want the raw draws — e.g. to compute both a covariance
    estimate and a shrinkage map without sampling twice.
    """
    X_tensor = _as_tensor(X, dtype=dtype)
    model = CorrShrinkCovModel(X_tensor, delta=delta, dtype=dtype)
    cfg = _resolve(preset, hmc_kwargs)
    samples = hmc_sample(model, **cfg)
    return samples, X_tensor


def fit_cov_ibb(X, preset="adhd_cov", method="plugin", **kwargs):
    """Estimate a covariance matrix with IBB.  Returns a numpy (d, d) array."""
    samples, X_tensor = run_chain(X, preset=preset, **kwargs)
    return reconstruct_sigma(samples, X_tensor, method=method).numpy()


def fit_corr_ibb(X, preset="adhd_cov", method="plugin", **kwargs):
    """Estimate a correlation matrix with IBB.  Returns a numpy (d, d) array."""
    samples, X_tensor = run_chain(X, preset=preset, **kwargs)
    return reconstruct_corr(samples, X_tensor, method=method).numpy()


def significant_ibb(X, preset="adhd_shrinkage", tol=0.05, **kwargs):
    """Posterior shrinkage map for ``X``.

    Thin wrapper around :func:`~ibb.posterior.significant_mask`; see it for the
    meaning of the returned probabilities.
    """
    samples, X_tensor = run_chain(X, preset=preset, **kwargs)
    return significant_mask(samples, X_tensor, tol=tol)
