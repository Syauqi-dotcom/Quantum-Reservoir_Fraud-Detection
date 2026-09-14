"""
Stage B - ambiguity router.

The ambiguous band is the "review zone": scores that are high enough to be
suspicious but not high enough to auto-decline.  Concretely, on the calibrated
validation scores p0:

    tau_low  = threshold achieving fraud recall  `recall_lo`  (e.g. 0.90)
               -> below this we approve (accept the ~10% missed fraud as the
                  operating point)
    tau_high = threshold achieving fraud recall  `recall_hi`  (e.g. 0.45)
               -> above this we auto-decline / send to manual review

Everything between is routed to QRC.  If that exceeds the routing budget the
band is shrunk symmetrically in log-odds space until it fits.  The band is then
FROZEN and applied once to the test set.
"""
from __future__ import annotations

import numpy as np

from .calibration import threshold_for_recall


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_band(y_val, p0_val, recall_lo: float, recall_hi: float, budget: float,
             min_pos: int = 20, y_train=None, p0_train=None):
    y_val = np.asarray(y_val)
    p0_val = np.asarray(p0_val, float)

    t_star = threshold_for_recall(y_val, p0_val, recall_lo)
    tau_low = t_star
    tau_high = threshold_for_recall(y_val, p0_val, recall_hi)
    if tau_high <= tau_low:
        tau_high = float(np.quantile(p0_val, 0.995))

    L = _logit(p0_val)
    lo, hi = _logit(tau_low), _logit(tau_high)
    mid = 0.5 * (lo + hi)

    routed = float(((L >= lo) & (L <= hi)).mean())
    shrink = 1.0
    while routed > budget and shrink > 0.05:
        shrink *= 0.9
        lo_s, hi_s = mid - (mid - lo) * shrink, mid + (hi - mid) * shrink
        routed = float(((L >= lo_s) & (L <= hi_s)).mean())
        lo, hi = lo_s, hi_s

    def _pos(a, b):
        n = int(y_val[(L >= a) & (L <= b)].sum())
        if p0_train is not None and y_train is not None:
            Lt = _logit(p0_train)
            n += int(np.asarray(y_train)[(Lt >= a) & (Lt <= b)].sum())
        return n

    # grow again (respecting nothing but min_pos) if we shrank too far
    while _pos(lo, hi) < min_pos and (hi - lo) < 8:
        lo -= 0.15
        hi += 0.15

    return dict(tau_low=float(_sig(lo)), tau_high=float(_sig(hi)),
               t_star=float(t_star), recall_lo=recall_lo, recall_hi=recall_hi,
               routed_fraction_val=float(((L >= lo) & (L <= hi)).mean()),
               pos_in_band_trainval=int(_pos(lo, hi)))


def apply_band(p0, band: dict):
    p0 = np.asarray(p0, float)
    return (p0 >= band["tau_low"]) & (p0 <= band["tau_high"])
