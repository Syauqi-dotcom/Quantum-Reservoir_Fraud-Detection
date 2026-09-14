"""Probability calibration (fit on validation only) + routing threshold search."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._m = None

    def fit(self, p_raw, y):
        p_raw = np.clip(np.asarray(p_raw, float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            self._m = IsotonicRegression(out_of_bounds="clip")
            self._m.fit(p_raw, y)
        else:                                            # platt
            self._m = LogisticRegression(max_iter=1000)
            self._m.fit(_logit(p_raw).reshape(-1, 1), y)
        return self

    def transform(self, p_raw):
        p_raw = np.clip(np.asarray(p_raw, float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self._m.predict(p_raw), 1e-6, 1 - 1e-6)
        return self._m.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def threshold_for_recall(y, p, recall_target: float) -> float:
    """Smallest threshold achieving >= recall_target on (y, p)."""
    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    p_sorted = np.asarray(p)[order]
    tp = np.cumsum(y_sorted)
    total_pos = y_sorted.sum()
    recall = tp / max(total_pos, 1)
    hit = np.searchsorted(recall, recall_target)
    hit = min(hit, len(p_sorted) - 1)
    return float(p_sorted[hit])


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins: int = 15):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)
