"""Hamiltonian Monte Carlo sampler for :class:`~ibb.model.CorrShrinkCovModel`.

A plain leapfrog HMC with a unit mass matrix.  It is written directly against
``torch.nn.Parameter`` objects rather than a functional log-density, which means
position updates must happen under ``no_grad`` while gradient evaluations happen
under ``enable_grad`` — mixing the two is the usual source of silent breakage
here, so the two phases are kept strictly apart below.

Non-finite proposals (a Cholesky failure, an overflowing quadratic form) are
rejected instead of aborting the chain.
"""

import torch


def hmc_sample(
    model,
    num_samples=500,
    burnin=200,
    thin=1,
    step_size=0.01,
    num_steps=20,
    seed=0,
    verbose=True,
    progress_every=50,
    return_diagnostics=False,
):
    """Draw samples from ``model.log_posterior()`` with leapfrog HMC.

    Parameters
    ----------
    model : CorrShrinkCovModel
        Any module exposing ``log_posterior()`` and ``draw_state()``.
    num_samples : int
        Number of retained draws (after burn-in and thinning).
    burnin : int
        Iterations discarded before collection starts.
    thin : int
        Keep every ``thin``-th post-burn-in iteration.
    step_size : float
        Leapfrog step size.  The model is stiff in high dimension — see the
        table in ``README.md`` for the values used per experiment.
    num_steps : int
        Leapfrog steps per proposal.
    seed : int
        Seed for the momentum draws and the accept/reject uniforms.
    verbose : bool
        Print log-posterior and running acceptance rate periodically.
    progress_every : int
        Iterations between progress lines.
    return_diagnostics : bool
        If True, return ``(samples, diagnostics)`` instead of just ``samples``.

    Returns
    -------
    list of dict
        Each entry is a ``model.draw_state()`` snapshot: ``theta``, ``sigma``,
        ``tau``.
    dict, optional
        Present when ``return_diagnostics``: ``accept_rate`` over all
        iterations, ``n_accepted``, ``n_iterations``, ``n_nonfinite`` (times the
        current state had no usable gradient), and the final ``log_posterior``.
        An acceptance rate far from ~0.6-0.9 means the step size needs retuning
        — near 0 the chain never moved, so the draws are all the initial point.
    """
    torch.manual_seed(seed)
    params = list(model.parameters())
    device, dtype = model.device, model.dtype

    def U_and_grads():
        """Potential energy U = -log_post, and its gradient w.r.t. params.

        Returns ``(U, None)`` when the log-posterior bailed out — a non-positive
        diagonal or a failed Cholesky makes it return a detached constant, which
        has no ``grad_fn`` to differentiate.  Callers treat a ``None`` gradient
        as "reject this proposal".
        """
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        lp = model.log_posterior()
        U = -lp
        if not U.requires_grad or not torch.isfinite(U):
            return U.detach(), None
        U.backward()
        grads = [p.grad.detach().clone() for p in params]
        return U.detach(), grads

    samples = []
    total = burnin + num_samples * thin
    accepts = 0
    nonfinite = 0
    finite_current = True

    for t in range(total):
        # Cache the current position so a rejected proposal can be undone.
        with torch.no_grad():
            q0 = [p.detach().clone() for p in params]

        # Current energy / gradient must be finite before we can move.
        with torch.enable_grad():
            U0, g0 = U_and_grads()
        if g0 is None or not torch.isfinite(U0):
            finite_current = False
            nonfinite += 1
            with torch.no_grad():
                for p, q in zip(params, q0):
                    p.copy_(q)
            continue
        finite_current = True

        # Momentum ~ N(0, I); kinetic energy of the starting state.
        p0 = [torch.randn_like(p) for p in params]
        K0 = 0.5 * sum(torch.sum(m * m) for m in p0)

        with torch.no_grad():
            q = [qq.clone() for qq in q0]
        # Half step for momentum.
        pcur = [m - 0.5 * step_size * g for m, g in zip(p0, g0)]

        proposed_ok = True
        U = None
        for lf in range(num_steps):
            # Full step for position — must not build a graph.
            with torch.no_grad():
                for i, par in enumerate(params):
                    q[i].add_(step_size * pcur[i])
                    par.copy_(q[i])

            with torch.enable_grad():
                U, g = U_and_grads()
            if g is None or not torch.isfinite(U):
                proposed_ok = False
                break

            # Full momentum step, except a half step on the last iteration.
            if lf != num_steps - 1:
                pcur = [m - step_size * gg for m, gg in zip(pcur, g)]
            else:
                pcur = [m - 0.5 * step_size * gg for m, gg in zip(pcur, g)]

        if (not proposed_ok) or (U is None):
            with torch.no_grad():
                for par, qq in zip(params, q0):
                    par.copy_(qq)
        else:
            K = 0.5 * sum(torch.sum(m * m) for m in pcur)
            log_alpha = (U0 + K0) - (U + K)

            # Compare in log space so a large energy gap cannot overflow.
            if torch.log(torch.rand((), device=device, dtype=dtype)) < log_alpha:
                accepts += 1
            else:
                with torch.no_grad():
                    for par, qq in zip(params, q0):
                        par.copy_(qq)

        if t >= burnin and ((t - burnin) % thin == 0) and finite_current:
            samples.append(model.draw_state())

        if verbose and (t + 1) % progress_every == 0:
            with torch.no_grad():
                lp = model.log_posterior()
                acc_rate = accepts / max(1, (t + 1))
                print(f"iter {t+1}/{total}  logpost={lp.item():.2f}  acc={acc_rate:.2f}")

    if not return_diagnostics:
        return samples

    with torch.no_grad():
        final_lp = float(model.log_posterior())

    return samples, {
        "accept_rate": accepts / max(1, total),
        "n_accepted": accepts,
        "n_iterations": total,
        "n_nonfinite": nonfinite,
        "log_posterior": final_lp,
    }
