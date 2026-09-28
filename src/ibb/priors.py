"""Log-densities of the priors used by the IBB shrinkage model.

The horseshoe prior on the shrinkage weights is built from two half-Cauchy
factors (a global scale ``tau`` and local scales ``lambda_j``); the marginal
variances carry an inverse-gamma prior.  Both densities are written in a form
that stays finite for very small / very large arguments, which matters because
HMC explores the tails of the log-scale parameterisation.
"""

import math

import torch


def log_half_cauchy(x, scale=1.0):
    """log density of a half-Cauchy(0, ``scale``) evaluated at ``x`` > 0.

    ``log1p((x/scale)**2)`` is used instead of ``log(1 + (x/scale)**2)`` so the
    term does not lose precision when ``x`` is much smaller than ``scale``.
    """
    return (
        math.log(2.0)
        - math.log(math.pi)
        - math.log(scale)
        - torch.log1p((x / scale) ** 2)
    )


def log_inv_gamma(x, alpha, beta):
    """log density of an InvGamma(``alpha``, ``beta``) evaluated at ``x`` > 0.

    Density: ``beta**alpha / Gamma(alpha) * x**(-alpha-1) * exp(-beta/x)``.
    """
    lgamma_alpha = torch.lgamma(torch.tensor(alpha, dtype=x.dtype, device=x.device))
    return (
        alpha * math.log(beta)
        - lgamma_alpha
        - (alpha + 1.0) * torch.log(x)
        - beta / x
    )
