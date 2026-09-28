"""Decoupled variant of the IBB model: the diagonal is a free parameter.

Motivation
----------
In :class:`~ibb.model.CorrShrinkCovModel` the diagonal is tied to the shrinkage
weights by construction::

    D = diag(1 - theta^2 + delta)

so driving ``theta -> 1`` (keep the full empirical correlation structure)
necessarily collapses ``D`` to the numerical floor ``delta``. The two cannot be
satisfied at once. Because the likelihood term ``-n/2 * log|Sigma|`` *rewards* a
near-singular matrix — and the data, living in an n-dimensional subspace of R^d,
never penalises it — the posterior mode sits at the ``theta -> 1`` boundary
regardless of the data. See the README section "The posterior mode is at
theta->1".

This variant breaks the coupling: the idiosyncratic variances become their own
parameter ``psi`` with an inverse-gamma prior, which keeps them away from zero::

    Sigma_corr = diag(psi) + U U^T,   U = diag(theta) Z / sqrt(n)

Everything else is unchanged — the horseshoe on ``theta``, the log-space
parameterisation, and the Woodbury/Cholesky evaluation all carry over verbatim,
since the matrix still has the ``diagonal + low-rank`` structure.

What this does and does not fix
-------------------------------
It removes the boundary degeneracy: the profiled posterior gains an interior
maximum. It does **not** remove the circularity — ``U`` is still built from the
data ``Z`` that the likelihood then scores, so the mode still sits at a larger
``theta`` (less shrinkage) than the risk-optimal one. Removing that requires
making the low-rank factor a free parameter too, i.e. a genuine Bayesian sparse
factor model.
"""

import math

import torch

from .priors import log_half_cauchy, log_inv_gamma


class DecoupledCorrShrinkCovModel(torch.nn.Module):
    """Horseshoe-shrinkage covariance model with a free idiosyncratic diagonal.

    Parameters
    ----------
    X : torch.Tensor
        Data matrix of shape (n, d).
    psi_alpha, psi_beta : float
        Shape / rate of the inverse-gamma prior on the diagonal ``psi``. The
        default ``InvGamma(3, 1)`` has mean 0.5 and mode 0.25, covering the
        ``(0, 1]`` range that ``1 - theta^2`` used to occupy — so the prior sits
        where the old deterministic diagonal lived, without pinning it there.
    tau0, ig_alpha, ig_beta, theta_eps
        As in :class:`~ibb.model.CorrShrinkCovModel`.

    Notes
    -----
    ``sigma`` (marginal scale) and ``psi`` (idiosyncratic share) both influence
    the diagonal of ``Sigma``, so they are only weakly separated by the
    likelihood; the off-diagonal identifies ``sigma``. In practice this shows up
    as a mildly ridged posterior. Watch the acceptance rate.
    """

    def __init__(
        self,
        X,
        theta_eps=1e-6,
        tau0=1.0,
        ig_alpha=3.0,
        ig_beta=1.0,
        psi_alpha=3.0,
        psi_beta=1.0,
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()
        device = device or X.device
        X = X.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype

        self.n, self.d = X.shape
        self.theta_eps = theta_eps
        self.tau0 = tau0
        self.ig_alpha = ig_alpha
        self.ig_beta = ig_beta
        self.psi_alpha = psi_alpha
        self.psi_beta = psi_beta

        Xc = X - X.mean(dim=0, keepdim=True)
        self.Xc = Xc

        s = Xc.std(dim=0, unbiased=False) + torch.tensor(1e-8, device=device, dtype=dtype)
        self.sample_std = s
        self.Z = (Xc / s).T.contiguous()  # (d, n)

        self.log_sigma2 = torch.nn.Parameter(torch.log(s ** 2))
        self.log_tau = torch.nn.Parameter(
            torch.tensor(-2.0, device=device, dtype=dtype)
        )
        self.log_lambda = torch.nn.Parameter(
            torch.zeros(self.d, device=device, dtype=dtype)
        )

        # NEW: the diagonal is its own parameter. Initialised at the value the
        # original model would have had at the same starting theta, so the two
        # chains begin from the same point and differences are attributable to
        # the decoupling rather than to a different start.
        u0 = math.exp(-2.0)
        theta0_sq = u0 ** 2 / (1.0 + u0 ** 2)
        self.log_psi = torch.nn.Parameter(
            torch.full((self.d,), math.log(1.0 - theta0_sq),
                       device=device, dtype=dtype)
        )

    def _transform(self):
        sigma2 = torch.exp(self.log_sigma2)
        sigma = torch.sqrt(sigma2)

        tau = torch.exp(self.log_tau)
        lam = torch.exp(self.log_lambda)
        u = tau * lam
        theta = u / torch.sqrt(1.0 + u ** 2)
        theta = torch.clamp(theta, self.theta_eps, 1.0 - self.theta_eps)

        psi = torch.exp(self.log_psi)
        return sigma2, sigma, tau, lam, theta, psi

    def log_posterior(self):
        """Unnormalised log-posterior; ``-inf`` on an invalid proposal."""
        sigma2, sigma, tau, lam, theta, psi = self._transform()

        diag_D = psi  # <-- the whole change: free, not 1 - theta^2 + delta
        if torch.any(diag_D <= 0):
            return torch.tensor(-torch.inf, device=self.device, dtype=self.dtype)

        inv_diag_D = 1.0 / diag_D

        U = (theta.unsqueeze(1) * self.Z) / math.sqrt(self.n)
        U_scaled = U * inv_diag_D.unsqueeze(1)

        M = torch.eye(self.n, device=self.device, dtype=self.dtype) + (U.T @ U_scaled)

        try:
            Lm = torch.linalg.cholesky(M)
        except RuntimeError:
            return torch.tensor(-torch.inf, device=self.device, dtype=self.dtype)

        logdet_D = torch.sum(torch.log(diag_D))
        logdet_M = 2.0 * torch.sum(torch.log(torch.diagonal(Lm)))
        logdet_Sigma = torch.sum(torch.log(sigma2)) + logdet_D + logdet_M

        Y = self.Xc / sigma.unsqueeze(0)
        tr_diag = torch.sum((Y ** 2) * inv_diag_D.unsqueeze(0))

        V = (Y @ U_scaled).T
        W = torch.cholesky_solve(V, Lm)
        quad = tr_diag - torch.sum(V * W)

        ll = -0.5 * self.n * logdet_Sigma - 0.5 * quad

        lp_sigma2 = torch.sum(
            log_inv_gamma(sigma2, self.ig_alpha, self.ig_beta) + torch.log(sigma2)
        )
        lp_tau = log_half_cauchy(tau, self.tau0) + torch.log(tau)
        lp_lam = torch.sum(log_half_cauchy(lam, 1.0) + torch.log(lam))
        # NEW: the prior that keeps psi off zero, which is what removes the
        # unbounded reward for a near-singular Sigma.
        lp_psi = torch.sum(
            log_inv_gamma(psi, self.psi_alpha, self.psi_beta) + torch.log(psi)
        )

        return ll + lp_sigma2 + lp_tau + lp_lam + lp_psi

    @torch.no_grad()
    def draw_state(self):
        """Snapshot of the interpretable parameters, detached and on the CPU."""
        _s2, sigma, tau, _lam, theta, psi = self._transform()
        return {
            "theta": theta.detach().cpu(),
            "sigma": sigma.detach().cpu(),
            "psi": psi.detach().cpu(),
            "tau": float(tau.detach().cpu()),
        }


def reconstruct_sigma_decoupled(samples, X, method="plugin", eps=1e-8,
                                dtype=torch.float64):
    """Rebuild a covariance estimate from draws of the decoupled model.

    Mirrors :func:`ibb.posterior.reconstruct_sigma`, but the residual diagonal
    is ``sigma^2 * psi`` rather than ``sigma^2 * (1 - theta^2)`` — the two
    models differ in exactly that term.
    """
    X = X.to(dtype=dtype)
    n, d = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    s = Xc.std(dim=0, unbiased=False) + eps
    Z = (Xc / s).T.contiguous()

    def one(theta, sigma, psi):
        resid = (sigma ** 2) * psi + eps
        V = (sigma * theta).unsqueeze(1) * Z / math.sqrt(n)
        return torch.diag(resid) + V @ V.T

    if method == "plugin":
        theta = torch.stack([x["theta"].to(dtype=dtype) for x in samples]).mean(0)
        sigma = torch.stack([x["sigma"].to(dtype=dtype) for x in samples]).mean(0)
        psi = torch.stack([x["psi"].to(dtype=dtype) for x in samples]).mean(0)
        return one(theta, sigma, psi)

    if method == "mean":
        Sig = torch.zeros((d, d), dtype=dtype)
        for x in samples:
            Sig += one(
                x["theta"].to(dtype=dtype),
                x["sigma"].to(dtype=dtype),
                x["psi"].to(dtype=dtype),
            )
        return Sig / len(samples)

    raise ValueError("method must be 'plugin' or 'mean'")
