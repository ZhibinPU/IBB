"""The IBB (Individual Bayesian Bagging) correlation-shrinkage covariance model.

The model treats the sample correlation structure as a low-rank object that is
shrunk towards the identity one coordinate at a time:

    x_i ~ N(0, Sigma),   Sigma = diag(sigma) @ Sigma_corr(theta) @ diag(sigma)

    Sigma_corr(theta) = D + U U^T,   D = diag(1 - theta^2 + delta)
                                     U = diag(theta) @ Z / sqrt(n)

``Z`` is the (d, n) column-standardised data matrix, held fixed.  Each ROI /
variable ``j`` gets its own shrinkage weight ``theta_j in (0, 1)``: at
``theta_j -> 0`` the variable is fully shrunk to independence, at
``theta_j -> 1`` it keeps its empirical correlations.  ``theta`` is built from a
horseshoe prior, so most weights are pulled to zero while a few can stay large.

Because ``Sigma_corr`` is a diagonal-plus-low-rank matrix, the log-likelihood is
evaluated with the Woodbury identity and a Cholesky factor of the (n, n) capacitance
matrix, i.e. at O(n^3 + n^2 d) instead of O(d^3).  That is what makes d >> n
feasible.
"""

import math

import torch

from .priors import log_half_cauchy, log_inv_gamma


class CorrShrinkCovModel(torch.nn.Module):
    """Horseshoe-shrinkage covariance model with a Woodbury log-posterior.

    Parameters
    ----------
    X : torch.Tensor
        Data matrix of shape (n, d) — n samples, d variables.
    delta : float
        Ridge added to the diagonal ``D`` so it stays strictly positive when
        ``theta`` approaches 1.
    theta_eps : float
        ``theta`` is clamped to ``[theta_eps, 1 - theta_eps]``.
    tau0 : float
        Scale of the half-Cauchy prior on the global shrinkage ``tau``.
    ig_alpha, ig_beta : float
        Shape / rate of the inverse-gamma prior on the marginal variances.

    Notes
    -----
    Sampling happens in log-space (``log_sigma2``, ``log_tau``, ``log_lambda``)
    so the parameters are unconstrained; :meth:`log_posterior` adds the
    corresponding Jacobian terms.
    """

    def __init__(
        self,
        X,
        delta=1e-4,
        theta_eps=1e-6,
        tau0=1.0,
        ig_alpha=3.0,
        ig_beta=1.0,
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()
        device = device or X.device
        X = X.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype

        self.n, self.d = X.shape
        self.delta = torch.tensor(delta, device=device, dtype=dtype)
        self.theta_eps = theta_eps
        self.tau0 = tau0
        self.ig_alpha = ig_alpha
        self.ig_beta = ig_beta

        # Center data
        Xc = X - X.mean(dim=0, keepdim=True)
        self.Xc = Xc

        # Standardize columns to build Z
        s = Xc.std(dim=0, unbiased=False) + torch.tensor(1e-8, device=device, dtype=dtype)
        self.sample_std = s
        self.Z = (Xc / s).T.contiguous()  # (d, n)

        # Parameterize with log(sigma^2) to get a clean IG prior + Jacobian
        self.log_sigma2 = torch.nn.Parameter(torch.log(s ** 2))  # (d,)
        self.log_tau = torch.nn.Parameter(
            torch.tensor(-2.0, device=device, dtype=dtype)
        )  # scalar
        self.log_lambda = torch.nn.Parameter(
            torch.zeros(self.d, device=device, dtype=dtype)
        )

    def _transform(self):
        """Map the unconstrained log-parameters to the model scale."""
        sigma2 = torch.exp(self.log_sigma2)  # (d,)
        sigma = torch.sqrt(sigma2)

        tau = torch.exp(self.log_tau)  # scalar > 0
        lam = torch.exp(self.log_lambda)  # (d,) > 0

        # u = tau * lambda is the horseshoe scale; the map u -> u/sqrt(1+u^2)
        # sends it into (0, 1) so it can act directly as a correlation weight.
        u = tau * lam
        theta = u / torch.sqrt(1.0 + u ** 2)
        theta = torch.clamp(theta, self.theta_eps, 1.0 - self.theta_eps)
        return sigma2, sigma, tau, lam, theta

    def log_posterior(self):
        """Unnormalised log-posterior at the current parameter values.

        Returns ``-inf`` (rather than raising) when the proposal leaves the
        valid region, so the HMC sampler can simply reject it.
        """
        sigma2, sigma, tau, lam, theta = self._transform()
        theta2 = theta ** 2

        diag_D = (1.0 - theta2) + self.delta
        if torch.any(diag_D <= 0):
            return torch.tensor(-torch.inf, device=self.device, dtype=self.dtype)

        inv_diag_D = 1.0 / diag_D

        # Low-rank factor U (d, n)
        U = (theta.unsqueeze(1) * self.Z) / math.sqrt(self.n)
        U_scaled = U * inv_diag_D.unsqueeze(1)  # D^{-1} U

        # Capacitance matrix M = I_n + U^T D^{-1} U   (n, n)
        M = torch.eye(self.n, device=self.device, dtype=self.dtype) + (U.T @ U_scaled)

        try:
            Lm = torch.linalg.cholesky(M)
        except RuntimeError:
            return torch.tensor(-torch.inf, device=self.device, dtype=self.dtype)

        # log|Sigma| = sum log(sigma^2) + log|D| + log|M|
        logdet_D = torch.sum(torch.log(diag_D))
        logdet_M = 2.0 * torch.sum(torch.log(torch.diagonal(Lm)))
        logdet_Sigma = torch.sum(torch.log(sigma2)) + logdet_D + logdet_M

        # Quadratic term via Woodbury:
        #   tr(Y Sigma_corr^{-1} Y^T) = tr(Y D^{-1} Y^T) - tr(V^T M^{-1} V)
        Y = self.Xc / sigma.unsqueeze(0)  # (n, d)
        tr_diag = torch.sum((Y ** 2) * inv_diag_D.unsqueeze(0))

        V = (Y @ U_scaled).T  # (n, n) = U^T D^{-1} Y^T
        W = torch.cholesky_solve(V, Lm)  # M^{-1} V
        tr_corr = torch.sum(V * W)
        quad = tr_diag - tr_corr

        ll = -0.5 * self.n * logdet_Sigma - 0.5 * quad

        # Priors + Jacobians of the log-space parameterisation.
        # sigma2 = exp(log_sigma2)  =>  |d sigma2 / d log_sigma2| = sigma2
        lp_sigma2 = torch.sum(
            log_inv_gamma(sigma2, self.ig_alpha, self.ig_beta) + torch.log(sigma2)
        )
        # tau ~ half-Cauchy(0, tau0)
        lp_tau = log_half_cauchy(tau, self.tau0) + torch.log(tau)
        # lambda_j ~ half-Cauchy(0, 1)
        lp_lam = torch.sum(log_half_cauchy(lam, 1.0) + torch.log(lam))

        return ll + lp_sigma2 + lp_tau + lp_lam

    @torch.no_grad()
    def draw_state(self):
        """Snapshot of the interpretable parameters, detached and on the CPU."""
        sigma2, sigma, tau, lam, theta = self._transform()
        return {
            "theta": theta.detach().cpu(),
            "sigma": sigma.detach().cpu(),
            "tau": float(tau.detach().cpu()),
        }
