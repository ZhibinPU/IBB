"""Downstream evaluation: does a better covariance estimate classify better?

The classifier is a plain quadratic discriminant — one mean and one covariance
per class, scored by the log posterior ratio.  Nothing is learned beyond those
estimates, so differences in AUC / F1 are attributable to the covariance
estimator rather than to a downstream model, which is the point of the
comparison.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from .stats import logpdf_mvn


def qda_scores(X_test, mu_pos, Sigma_pos, mu_neg, Sigma_neg, pi_pos, jitter=1e-3):
    """Log posterior ratio ``log p(x | pos) + log pi_pos - (neg terms)``.

    Positive values favour the positive class.  Returned as a continuous score
    so it can feed either a ROC curve or a hard threshold at 0.
    """
    pi_neg = 1.0 - pi_pos
    scores = []
    for x in X_test:
        lp = logpdf_mvn(x, mu_pos, Sigma_pos, jitter=jitter) + np.log(pi_pos)
        ln = logpdf_mvn(x, mu_neg, Sigma_neg, jitter=jitter) + np.log(pi_neg)
        scores.append(lp - ln)
    return np.asarray(scores)


def cross_validate_qda(
    X,
    y,
    fit_cov,
    n_splits=3,
    n_repeats=5,
    random_state=1,
    jitter=1e-3,
    center=True,
    verbose=False,
):
    """Repeated stratified CV of a QDA built on ``fit_cov``.

    The covariance estimator is re-fitted per class *inside* every fold, so no
    test-fold information reaches the estimate.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    y : ndarray, shape (n,)
        Binary labels; ``1`` is the positive class.
    fit_cov : callable
        ``fit_cov(X_class) -> (d, d)``.  E.g. ``ibb.fit_cov_ibb`` or
        ``ibb.baselines.fit_cov_bag``.
    center : bool
        Subtract the overall column mean before splitting.  Matches the original
        experiments; it is a per-dataset constant shift, identical across folds.

    Returns
    -------
    dict of ndarray
        Per-fold ``auc``, ``f1``, ``accuracy``, ``precision``, ``recall``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if center:
        X = X - np.mean(X, axis=0)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    out = {k: [] for k in ("auc", "f1", "accuracy", "precision", "recall")}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_pos = X_tr[y_tr == 1]
        X_neg = X_tr[y_tr == 0]

        if verbose:
            print(
                f"fold {fold + 1}: train={len(X_tr)} "
                f"(pos={len(X_pos)}, neg={len(X_neg)}), test={len(X_te)}"
            )

        mu_pos = X_pos.mean(axis=0)
        mu_neg = X_neg.mean(axis=0)

        Sigma_pos = fit_cov(X_pos)
        Sigma_neg = fit_cov(X_neg)

        pi_pos = len(X_pos) / len(X_tr)

        scores = qda_scores(
            X_te, mu_pos, Sigma_pos, mu_neg, Sigma_neg, pi_pos, jitter=jitter
        )
        y_pred = (scores >= 0).astype(int)

        out["auc"].append(roc_auc_score(y_te, scores))
        out["f1"].append(f1_score(y_te, y_pred, zero_division=0))
        out["accuracy"].append(accuracy_score(y_te, y_pred))
        out["precision"].append(precision_score(y_te, y_pred, zero_division=0))
        out["recall"].append(recall_score(y_te, y_pred, zero_division=0))

    return {k: np.asarray(v) for k, v in out.items()}


def summarise(results):
    """Mean and sd of every metric in a :func:`cross_validate_qda` result."""
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}
