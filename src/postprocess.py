"""
Stage H - entity-level prediction smoothing.

Motivation: the IEEE-CIS Kaggle 1st-place solution's biggest single lever
(beyond feature engineering) was a one-line post-processing trick - replace
every transaction's score with its card/client's mean score across the
evaluation batch. That is a real, cheap, classical improvement independent of
the quantum question this PoC is otherwise about, so it deserves its own
honest test.

Two variants, because the Kaggle trick as originally described is NOT
deployable in HSBC's real-time authorization path:

  causal_smooth  : REAL-TIME SAFE. For a transaction at time t, blend its own
                   score with the EXPANDING mean of that same entity's PAST
                   scores only (t' < t). No future information is used, so
                   this could run in production exactly as-is.

  batch_smooth   : the exact Kaggle trick - replace every transaction's score
                   with its entity's mean score across the WHOLE evaluation
                   batch (train+val+test all mixed within each set). This
                   uses "future" transactions of the same entity relative to
                   any given transaction, which is fine for a one-shot batch
                   Kaggle submission but is NOT something HSBC's <300ms
                   authorization flow could do (you do not have the entity's
                   future transactions yet). Reported only as an offline /
                   after-the-fact audit ceiling, never as a deployable number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def causal_smooth(df_time_sorted: pd.DataFrame, p_col: str, entity_col: str,
                  alpha: float) -> np.ndarray:
    """alpha=1.0 -> raw score unchanged; alpha=0.0 -> fully replaced by the
    entity's expanding past-only mean. First occurrence of an entity is
    always left as its raw score (no past to blend with)."""
    p = df_time_sorted[p_col].to_numpy(dtype=float)
    ent = df_time_sorted[entity_col].to_numpy()
    out = np.empty(len(p))
    run_sum: dict = {}
    run_cnt: dict = {}
    for i in range(len(p)):
        k = ent[i]
        cnt = run_cnt.get(k, 0)
        if cnt > 0:
            past_mean = run_sum[k] / cnt
            out[i] = alpha * p[i] + (1 - alpha) * past_mean
        else:
            out[i] = p[i]
        run_sum[k] = run_sum.get(k, 0.0) + p[i]
        run_cnt[k] = cnt + 1
    return out


def batch_smooth(df: pd.DataFrame, p_col: str, entity_col: str) -> np.ndarray:
    """Non-causal Kaggle-style trick: mean score per entity within this batch."""
    return df.groupby(entity_col)[p_col].transform("mean").to_numpy(dtype=float)


def select_alpha(df_full_sorted: pd.DataFrame, p_col: str, entity_col: str,
                 y_col: str, val_mask: np.ndarray, alpha_grid) -> float:
    """Freeze alpha on validation AUPRC only, before it ever touches test."""
    from sklearn.metrics import average_precision_score
    y_va = df_full_sorted.loc[val_mask, y_col].to_numpy()
    best = None
    for a in alpha_grid:
        smoothed = causal_smooth(df_full_sorted, p_col, entity_col, a)
        ap = average_precision_score(y_va, smoothed[val_mask])
        if best is None or ap > best[0]:
            best = (ap, a)
    return best[1]
