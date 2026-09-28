"""Turning HMC draws back into covariance / correlation matrices.

Each draw stores only ``theta`` and ``sigma`` (see
:meth:`~ibb.model.CorrShrinkCovModel.draw_state`); the low-rank factor is
rebuilt from the fixed standardised data ``Z``, so a full (d, d) matrix is only
materialised when it is actually needed.
"""

import math

import torch


def _standardised_factor(X, eps=1e-8, dtype=torch.float64):
    """Return ``(n, d, Z)`` where ``Z`` is the (d, n) column-standardised data."""
    X = X.to(dtype=dtype)
    n, d = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    s = Xc.std(dim=0, unbiased=False) + eps
    Z = (Xc / s).T.contiguous()  # (d, n)
    return n, d, Z


def reconstruct_sigma(samples, X, method="plugin", eps=1e-8, dtype=torch.float64):
    """Rebuild a covariance estimate from a list of HMC draws.

    Parameters
    ----------
    samples : list of dict
        Output of :func:`~ibb.hmc.hmc_sample`.
    X : torch.Tensor
        The (n, d) data the chain was run on.
    method : {"plugin", "mean"}
        ``"plugin"`` averages ``theta`` and ``sigma`` first and then forms a
        single ``Sigma`` — cheap, and what the experiments use.  ``"mean"`` is
        the honest posterior mean: it forms one ``Sigma`` per draw and averages
        the matrices, at O(S d^2) memory traffic.

    Returns
    -------
    torch.Tensor
        (d, d) covariance estimate.
    """
    n, d, Z = _standardised_factor(X, eps=eps, dtype=dtype)

    if method == "plugin":
        theta = torch.stack([smp["theta"].to(dtype=dtype) for smp in samples]).mean(0)
        sigma = torch.stack([smp["sigma"].to(dtype=dtype) for smp in samples]).mean(0)

        resid = (sigma ** 2) * (1.0 - theta ** 2) + eps
        V = (sigma * theta).unsqueeze(1) * Z / math.sqrt(n)
        return torch.diag(resid) + V @ V.T

    if method == "mean":
        Sig = torch.zeros((d, d), dtype=dtype)
        for smp in samples:
            theta = smp["theta"].to(dtype=dtype)
            sigma = smp["sigma"].to(dtype=dtype)
            resid = (sigma ** 2) * (1.0 - theta ** 2) + eps
            V = (sigma * theta).unsqueeze(1) * Z / math.sqrt(n)
            Sig += torch.diag(resid) + V @ V.T
        return Sig / len(samples)

    raise ValueError("method must be 'plugin' or 'mean'")


def reconstruct_corr(samples, X, method="plugin", eps=1e-8, dtype=torch.float64):
    """Same as :func:`reconstruct_sigma`, rescaled to a correlation matrix."""
    Sigma = reconstruct_sigma(samples, X, method=method, eps=eps, dtype=dtype)
    inv_sd = torch.diag(1.0 / torch.diag(Sigma)) ** 0.5
    return inv_sd @ Sigma @ inv_sd


def significant_mask(samples, X, tol=0.05, eps=1e-8, dtype=torch.float64):
    """Posterior shrinkage map: how often each entry moves away from the SCM.

    For every draw the model correlation ``Corr`` is compared entrywise against
    the sample correlation ``sample_cor``.  ``posterior_prob_geq[i, j]`` is the
    fraction of draws for which ``|Corr_ij - sample_cor_ij| > tol``, i.e. the
    posterior probability that entry (i, j) is *actively shrunk* rather than
    left at its empirical value.  ``posterior_prob_leq`` is its complement.

    Parameters
    ----------
    samples : list of dict
        Output of :func:`~ibb.hmc.hmc_sample`.
    X : torch.Tensor
        The (n, d) data the chain was run on.
    tol : float
        Deviation from the sample correlation that counts as shrinkage.  The
        simulation study uses 0.04, the ADHD-200 application 0.05.

    Returns
    -------
    dict
        ``posterior_prob_geq``, ``posterior_prob_leq``, ``mean_corr``, plus
        ``posterior_prob`` as an alias of ``posterior_prob_geq``.
    """
    n, d, Z = _standardised_factor(X, eps=eps, dtype=dtype)
    X = X.to(dtype=dtype)

    sample_cor = torch.corrcoef(X.T)
    count_geq = torch.zeros((d, d), dtype=dtype)
    count_leq = torch.zeros((d, d), dtype=dtype)
    mean_corr = torch.zeros((d, d), dtype=dtype)

    n_draws = 0

    for smp in samples:
        theta = smp["theta"].to(dtype=dtype)
        sigma = smp["sigma"].to(dtype=dtype)

        resid = (sigma ** 2) * (1.0 - theta ** 2) + eps
        V = (sigma * theta).unsqueeze(1) * Z / math.sqrt(n)
        Sigma = torch.diag(resid) + V @ V.T

        # Covariance -> correlation
        sd = torch.sqrt(torch.diag(Sigma) + eps)
        Corr = Sigma / (sd[:, None] * sd[None, :])

        dev = torch.abs(Corr - sample_cor)
        count_geq += (dev > tol).to(dtype)
        count_leq += (dev <= tol).to(dtype)
        mean_corr += Corr

        n_draws += 1

    if n_draws == 0:
        raise ValueError("`samples` is empty — the chain produced no draws.")

    posterior_prob_geq = count_geq / n_draws
    posterior_prob_leq = count_leq / n_draws
    mean_corr /= n_draws

    return {
        "posterior_prob_geq": posterior_prob_geq,
        "posterior_prob_leq": posterior_prob_leq,
        "posterior_prob": posterior_prob_geq,  # alias used by the ADHD scripts
        "mean_corr": mean_corr,
    }
