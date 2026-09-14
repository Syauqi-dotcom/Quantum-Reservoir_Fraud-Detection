"""Metrics + paired bootstrap for the frozen success criteria."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

from .calibration import threshold_for_recall, brier, ece


def basic_metrics(y, p, threshold=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        auprc=float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        f1=float(f1_score(y, pred, zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        brier=brier(y, p),
        ece=ece(y, p),
        confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        threshold=float(threshold),
        n=int(len(y)),
        n_pos=int(y.sum()),
    )


def false_decline_rate(y, p, recall_target, threshold=None):
    """Fraction of legitimate transactions declined at a threshold set for a
    given fraud recall (proxy for real false-decline cost)."""
    y = np.asarray(y)
    p = np.asarray(p)
    if threshold is None:
        threshold = threshold_for_recall(y, p, recall_target)
    pred = (p >= threshold).astype(int)
    legit = y == 0
    return float(pred[legit].mean()), float(threshold)


def paired_bootstrap_delta(y, p_a, p_b, metric_fn, iters=2000, ci=0.95, seed=0):
    """delta = metric(p_a) - metric(p_b) with a paired bootstrap CI."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    n = len(y)
    deltas = np.empty(iters)
    base = metric_fn(y, p_a) - metric_fn(y, p_b)
    for k in range(iters):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            deltas[k] = np.nan
            continue
        deltas[k] = metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx])
    lo = np.nanpercentile(deltas, 100 * (1 - ci) / 2)
    hi = np.nanpercentile(deltas, 100 * (1 + ci) / 2)
    return dict(delta=float(base), ci_low=float(lo), ci_high=float(hi),
               p_gt0=float(np.nanmean(deltas > 0)))


def auprc_fn(y, p):
    return average_precision_score(y, p) if np.asarray(y).sum() > 0 else np.nan


def fdr_fn_factory(recall_target):
    def _fn(y, p):
        return false_decline_rate(y, p, recall_target)[0]
    return _fn
