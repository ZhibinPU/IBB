"""Generative sparse factor model — version 09082144.

Third attempt at a well-posed version of the IBB idea. The two earlier ones and
why they failed:

* ``model.py`` — ``Sigma_corr = diag(1-theta^2+delta) + diag(theta) Z Z^T
  diag(theta)/n``. The diagonal is tied to ``theta``, so ``theta -> 1``
  collapses it to the numerical floor and the likelihood term ``-n/2 log|Sigma|``
  runs away. Posterior mode at the boundary, always.
* ``model_decoupled.py`` — frees the diagonal. Slows the climb to the boundary
  but does not stop it, because the low-rank factor is *still built from the
  data* and *still has rank n*.

What changes here
-----------------
Two things, and the second is the one that matters:

1. **The factor is a free parameter.** ``Lambda`` (d, k) is sampled, not
   constructed from ``Z``. So ``p(X | Sigma)`` is a genuine likelihood and
   ``Sigma`` has a prior that does not depend on the data — the model is
   generative, and the circularity is gone.

2. **The rank is capped at k << n.** With rank n the previous models could
   reproduce the sample covariance exactly, so the likelihood always preferred
   to. With k << n they structurally cannot, and the residual must be absorbed
   by ``Psi`` — which is what forces a genuine bias/variance trade-off.

Model::

    x_i ~ N(0, Sigma),   Sigma = Lambda Lambda^T + Psi,   Psi = diag(psi)

    Lambda_jl = tau * lambda_jl * z_jl     (non-centred horseshoe)
    lambda_jl ~ C+(0, 1),  tau ~ C+(0, tau0),  z_jl ~ N(0, 1)
    psi_j ~ InvGamma(a, b)

The horseshoe now sits on the *loadings*, which is where it belongs: it
sparsifies which variable loads on which factor, rather than scaling whole rows
of a correlation matrix. That also lifts the rank-1 restriction of the earlier
models, where the shrinkage factor was forced to be ``outer(theta, theta)`` and
an edge's fate was decided entirely by its two endpoints.

Cost
----
Woodbury now goes through a ``k x k`` capacitance matrix instead of ``n x n``,
so each evaluation is *cheaper* than before: O(k^3 + k^2 d) with k ~ 10.
Against that, the parameter count rises from ~d to ~(k+2)d.

Caveat
------
Factor models are rotationally non-identified: ``Lambda -> Lambda Q`` for
orthogonal ``Q`` leaves ``Sigma`` unchanged. ``Sigma`` itself — the only thing
reported — is identified, but the flat directions can slow mixing. Do not
interpret individual loadings without fixing a rotation.
"""

import math

import torch

from .priors import log_half_cauchy, log_inv_gamma


class SparseFactorCovModel(torch.nn.Module):
    """Bayesian sparse factor covariance model with horseshoe loadings.

    Parameters
    ----------
    X : torch.Tensor
        Data matrix of shape (n, d).
    n_factors : int
        Latent dimension ``k``. Must be ``< n`` for the model to be unable to
        fit the data exactly; the constructor warns if it is not comfortably so.
    tau0 : float
        Scale of the half-Cauchy prior on the global loading scale.
    psi_alpha, psi_beta : float
        Inverse-gamma prior on the idiosyncratic variances.
    init : {"pca", "random"}
        Starting point for the loadings. ``"pca"`` starts from the scaled top-k
        principal directions (fast, and a legitimate initialisation — it seeds
        the chain, it does not enter the model). ``"random"`` starts small and
        random; use it to check that the answer does not depend on the start.
    """

    def __init__(
        self,
        X,
        n_factors=10,
        tau0=1.0,
        psi_alpha=3.0,
        psi_beta=1.0,
        init="pca",
        seed=0,
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()
        device = device or X.device
        X = X.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype

        self.n, self.d = X.shape
        self.k = int(n_factors)
        self.tau0 = tau0
        self.psi_alpha = psi_alpha
        self.psi_beta = psi_beta

        if self.k >= self.n:
            raise ValueError(
                f"n_factors={self.k} must be < n={self.n}; with k >= n the model "
                "can reproduce the sample covariance exactly, which is the "
                "failure mode this version exists to avoid"
            )

        self.Xc = X - X.mean(dim=0, keepdim=True)
        s = self.Xc.std(dim=0, unbiased=False) + torch.tensor(
            1e-8, device=device, dtype=dtype
        )
        self.sample_std = s

        g = torch.Generator(device="cpu").manual_seed(seed)

        # --- loadings, non-centred horseshoe ---
        if init == "pca":
            # Top-k right singular vectors, scaled by their singular values.
            # Seeds the chain near a sensible basin; not part of the model.
            with torch.no_grad():
                _U, S, Vh = torch.linalg.svd(
                    self.Xc / math.sqrt(self.n), full_matrices=False
                )
                z0 = (Vh[: self.k].T * S[: self.k]).contiguous()
                z0 = z0 / (z0.std() + 1e-12) * 0.5
        elif init == "random":
            z0 = 0.1 * torch.randn(
                self.d, self.k, generator=g, dtype=dtype
            ).to(device)
        else:
            raise ValueError("init must be 'pca' or 'random'")

        self.z = torch.nn.Parameter(z0.to(device=device, dtype=dtype))
        self.log_lambda = torch.nn.Parameter(
            torch.zeros(self.d, self.k, device=device, dtype=dtype)
        )
        self.log_tau = torch.nn.Parameter(
            torch.tensor(-1.0, device=device, dtype=dtype)
        )

        # --- idiosyncratic variances, on the data's scale ---
        self.log_psi = torch.nn.Parameter(
            torch.log(0.5 * s ** 2).to(device=device, dtype=dtype)
        )

    def _transform(self):
        tau = torch.exp(self.log_tau)
        lam = torch.exp(self.log_lambda)
        Lambda = tau * lam * self.z          # (d, k)
        psi = torch.exp(self.log_psi)        # (d,)
        return Lambda, psi, tau, lam

    def log_posterior(self):
        """Unnormalised log-posterior; ``-inf`` on an invalid proposal."""
        Lambda, psi, tau, lam = self._transform()

        if torch.any(psi <= 0) or not torch.isfinite(Lambda).all():
            return torch.tensor(-torch.inf, device=self.device, dtype=self.dtype)

        inv_psi = 1.0 / psi

        # Capacitance matrix M = I_k + Lambda^T Psi^-1 Lambda   (k, k)
        Lam_scaled = Lambda * inv_psi.unsqueeze(1)
        M = torch.eye(self.k, device=self.device, dtype=self.dtype) + (
            Lambda.T @ Lam_scaled
        )

        try:
            Lm = torch.linalg.cholesky(M)
        except RuntimeError:
            return torch.tensor(-torch.inf, device=self.device, dtype=self.dtype)

        # log|Sigma| = log|Psi| + log|M|
        logdet_Sigma = torch.sum(torch.log(psi)) + 2.0 * torch.sum(
            torch.log(torch.diagonal(Lm))
        )

        # tr(Xc Sigma^-1 Xc^T) by Woodbury
        tr_diag = torch.sum((self.Xc ** 2) * inv_psi.unsqueeze(0))
        A = self.Xc @ Lam_scaled                       # (n, k)
        W = torch.cholesky_solve(A.T, Lm)              # (k, n)
        quad = tr_diag - torch.sum(A.T * W)

        ll = -0.5 * self.n * logdet_Sigma - 0.5 * quad

        # --- priors ---
        # z ~ N(0,1), the non-centred part of the horseshoe
        lp_z = -0.5 * torch.sum(self.z ** 2)
        # local scales, half-Cauchy, with the log-space Jacobian
        lp_lam = torch.sum(log_half_cauchy(lam, 1.0) + torch.log(lam))
        # global scale
        lp_tau = log_half_cauchy(tau, self.tau0) + torch.log(tau)
        # idiosyncratic variances
        lp_psi = torch.sum(
            log_inv_gamma(psi, self.psi_alpha, self.psi_beta) + torch.log(psi)
        )

        return ll + lp_z + lp_lam + lp_tau + lp_psi

    @torch.no_grad()
    def draw_state(self):
        """Snapshot of the interpretable quantities, detached and on the CPU."""
        Lambda, psi, tau, _lam = self._transform()
        return {
            "Lambda": Lambda.detach().cpu(),
            "psi": psi.detach().cpu(),
            "tau": float(tau.detach().cpu()),
        }

    @torch.no_grad()
    def communality(self):
        """Per-variable share of variance explained by the factors.

        ``diag(Lambda Lambda^T) / (diag(Lambda Lambda^T) + psi)`` — the direct
        analogue of ``theta^2`` in the earlier models, and the quantity to watch
        for a boundary run: 1.0 means the factors absorb everything and ``psi``
        has collapsed.
        """
        Lambda, psi, _tau, _lam = self._transform()
        comm = torch.sum(Lambda ** 2, dim=1)
        return (comm / (comm + psi)).detach().cpu()


def reconstruct_sigma_factor(samples, method="plugin", dtype=torch.float64):
    """Rebuild a covariance estimate from draws of the factor model.

    Unlike the earlier reconstructions this needs no reference to ``X`` — the
    draws already contain everything, which is precisely the point of a
    generative parameterisation.
    """
    if not samples:
        raise ValueError("`samples` is empty — the chain produced no draws.")

    if method == "plugin":
        Lambda = torch.stack([s["Lambda"].to(dtype=dtype) for s in samples]).mean(0)
        psi = torch.stack([s["psi"].to(dtype=dtype) for s in samples]).mean(0)
        return Lambda @ Lambda.T + torch.diag(psi)

    if method == "mean":
        d = samples[0]["psi"].shape[0]
        Sig = torch.zeros((d, d), dtype=dtype)
        for s in samples:
            L = s["Lambda"].to(dtype=dtype)
            Sig += L @ L.T + torch.diag(s["psi"].to(dtype=dtype))
        return Sig / len(samples)

    raise ValueError("method must be 'plugin' or 'mean'")
