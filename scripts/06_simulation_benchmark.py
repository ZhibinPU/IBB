#!/usr/bin/env python3
"""Simulation study: estimation error of IBB vs the sample covariance and bagging.

For each replicate a population covariance ``Sigma`` and data ``X`` (n, d) are
drawn from a known structure, every estimator is fitted, and the error is
recorded in both the Frobenius and spectral norms.  Timings are recorded too,
since the methods differ by orders of magnitude in cost.

Requires ``ibb.simulation`` (pending — see README, "Pending inputs").

Usage
-----
    python scripts/06_simulation_benchmark.py --structure block --n 300 --reps 5
"""

import argparse
import time

import numpy as np

from ibb.api import fit_cov_ibb
from ibb.baselines import fit_cov_bag, fit_cov_pot
from ibb.config import RESULTS_DIR, ensure_dirs
from ibb.stats import frobenius, spectral

STRUCTURES = {
    "block": ("simulation_block", {200: "sim_block_n200",
                                   300: "sim_block_n300",
                                   400: "sim_block_n400"}),
    "toeplitz": ("simulation_toeplitz", {}),
    "band": ("simulation_band", {}),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure", choices=sorted(STRUCTURES), default="block")
    ap.add_argument("--n", type=int, default=300, help="number of samples")
    ap.add_argument("--d", type=int, default=None,
                    help="number of variables (default: 15 * n, as in the paper)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--preset", default=None,
                    help="HMC preset; defaults to the one tuned for this (structure, n)")
    ap.add_argument("--poet-k", type=int, default=10, help="number of POET factors")
    ap.add_argument("--quiet", action="store_true", help="suppress HMC progress lines")
    args = ap.parse_args()

    ensure_dirs()

    d = args.d if args.d is not None else 15 * args.n
    gen_name, preset_by_n = STRUCTURES[args.structure]

    preset = args.preset or preset_by_n.get(args.n) or "sim_toeplitz"
    print(f"structure={args.structure}  n={args.n}  d={d}  preset={preset}")

    import ibb.simulation as sim
    generator = getattr(sim, gen_name)

    methods = ("scm", "bag", "pot", "ibb")
    errors = {f"{m}_{norm}": [] for m in methods for norm in ("fro", "spec")}
    timings = {m: [] for m in methods}

    for rep in range(args.reps):
        print(f"\n=== replicate {rep + 1}/{args.reps} ===")
        X, Sigma = generator(n=args.n, d=d, random_state=rep)

        estimates = {}

        t0 = time.time()
        estimates["scm"] = np.cov(X.T, ddof=1)
        timings["scm"].append(time.time() - t0)

        t0 = time.time()
        estimates["bag"] = fit_cov_bag(X)
        timings["bag"].append(time.time() - t0)

        t0 = time.time()
        estimates["pot"] = fit_cov_pot(X, K=args.poet_k)
        timings["pot"].append(time.time() - t0)

        t0 = time.time()
        estimates["ibb"] = fit_cov_ibb(X, preset=preset, verbose=not args.quiet)
        timings["ibb"].append(time.time() - t0)

        for m, est in estimates.items():
            errors[f"{m}_fro"].append(frobenius(est, Sigma))
            errors[f"{m}_spec"].append(spectral(est, Sigma))
            print(f"  {m:4s}  fro={errors[f'{m}_fro'][-1]:.3f}  "
                  f"spec={errors[f'{m}_spec'][-1]:.3f}  "
                  f"time={timings[m][-1]:.1f}s")

    print("\n=== summary (mean +/- sd over replicates) ===")
    for key, vals in errors.items():
        print(f"{key:10s}  {np.mean(vals):12.4f} +/- {np.std(vals):.4f}")
    for m, vals in timings.items():
        print(f"{m}_time    {np.mean(vals):12.2f}s")

    tag = f"{args.structure}_n{args.n}_d{d}"
    out = RESULTS_DIR / f"simulation_{tag}.npz"
    np.savez(out, **{k: np.asarray(v) for k, v in {**errors, **{
        f"{m}_time": t for m, t in timings.items()}}.items()})
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
