"""
Stage F - distillation for a real-time classical path.

The QPU cannot sit in the 100-300 ms authorisation window.  We train a small
classical student on the ambiguous-band rows to mimic the hybrid's final
probability p_final from *classical* inputs only (static + rolling + context),
then measure the metric gap and the inference latency.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:                                   # pragma: no cover
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def train_student(X_tr, p_final_tr, X_te, cfg_distill, seed):
    """Regress logit(p_final) from classical features; return probs on X_te + timing."""
    y = _logit(p_final_tr)
    if HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=cfg_distill.get("n_estimators", 200),
            num_leaves=cfg_distill.get("num_leaves", 16),
            max_depth=cfg_distill.get("max_depth", 4),
            learning_rate=0.05, random_state=seed, n_jobs=4, verbose=-1,
        )
    else:                                            # pragma: no cover
        model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X_tr, y)

    t0 = time.perf_counter()
    raw = model.predict(X_te)
    dt = time.perf_counter() - t0
    p = 1 / (1 + np.exp(-raw))
    per_row_ms = 1000 * dt / max(len(X_te), 1)
    return p, per_row_ms
