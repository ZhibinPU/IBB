#!/usr/bin/env python3
"""Downstream check: does a better covariance estimate classify better?

Repeated stratified CV of a quadratic discriminant whose per-class covariance
comes from IBB, bagging, or POET.  Nothing else is fitted, so the metric
differences are attributable to the covariance estimator.

Runs on the ADHD case/control tables by default.  ``--source synthetic``
instead draws heavy-tailed synthetic groups (requires the pending
``cov_ppca``).

Usage
-----
    python scripts/07_classification_benchmark.py --method ibb
    python scripts/07_classification_benchmark.py --method bag --source synthetic
"""

import argparse
import functools
import sys

import numpy as np

from ibb.api import fit_cov_ibb
from ibb.baselines import fit_cov_bag
from ibb.config import RESULTS_DIR, ensure_dirs
from ibb.evaluation import cross_validate_qda, summarise
from ibb_adhd.data import load_groups, stack_groups


def build_estimator(method, preset):
    if method == "ibb":
        return functools.partial(fit_cov_ibb, preset=preset, verbose=False)
    if method == "bag":
        return fit_cov_bag
    if method == "pot":
        from ibb.baselines.poet import fit_cov_pot

        return fit_cov_pot
    raise ValueError(method)


def load_source(source, n_synth, seed):
    if source == "adhd":
        X_case, X_control = load_groups()
        return stack_groups(X_case, X_control)

    # synthetic: heavy-tailed draws around a PPCA covariance of each group
    from ibb.synthetic import nonG_sim

    X_case, X_control = load_groups()
    syn_case, _ = nonG_sim(X_case, n=n_synth, random_state=seed)
    syn_control, _ = nonG_sim(X_control, n=n_synth, random_state=seed)
    return stack_groups(syn_case, syn_control)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=("ibb", "bag", "pot"), default="ibb")
    ap.add_argument("--source", choices=("adhd", "synthetic"), default="adhd")
    ap.add_argument("--preset", default="ct_real",
                    help="HMC preset for --method ibb")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-synth", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ensure_dirs()

    try:
        X, y = load_source(args.source, args.n_synth, args.seed)
        fit_cov = build_estimator(args.method, args.preset)
    except NotImplementedError as exc:
        sys.exit(f"{exc}")

    print(f"method={args.method}  source={args.source}  X={X.shape}  "
          f"pos={int(y.sum())}  neg={int((1 - y).sum())}")

    results = cross_validate_qda(
        X,
        y,
        fit_cov=fit_cov,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
        verbose=True,
    )

    print("\n=== summary (mean +/- sd over folds) ===")
    for metric, (mean, sd) in summarise(results).items():
        print(f"{metric:10s}  {mean:.4f} +/- {sd:.4f}")

    out = RESULTS_DIR / f"classification_{args.source}_{args.method}.npz"
    np.savez(out, **results)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
