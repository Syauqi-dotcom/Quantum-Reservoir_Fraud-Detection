"""
Stage C - sequence builder.

For a target transaction at time t belonging to entity_key k, collect up to K
transactions of k with time < t (plus the target itself as the last step),
build a per-step feature vector, left-pad with a mask, and min-max scale each
feature to [0, 1] with a scaler fit on the TRAIN sequences only (needed for
angle encoding into the quantum circuit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP_FEATURES = [
    "amt_z_entity", "amt_ratio_median", "dlog_time", "device_changed",
    "addr_changed", "email_risk", "burst_24h", "hist_score",
]


def build_sequences(df_all: pd.DataFrame, rolling: pd.DataFrame, p0_all: dict,
                    target_ids, K: int):
    """
    df_all   : full time-sorted frame with entity_key, TransactionID, TransactionDT
    rolling  : rolling-feature frame indexed by TransactionID
    p0_all   : dict TransactionID -> calibrated LightGBM score (for hist_score)
    returns  : seq (n, K, F) float32, mask (n, K) float32, lengths (n,)
    """
    df = df_all.sort_values("TransactionDT").reset_index(drop=True)
    by_ent = {k: g["TransactionID"].to_numpy() for k, g in df.groupby("entity_key")}
    tid_to_row = {t: i for i, t in enumerate(df["TransactionID"].to_numpy())}
    ent_of = dict(zip(df["TransactionID"], df["entity_key"]))

    roll = rolling
    email_risk = df.set_index("TransactionID")["email_risk"]

    F = len(STEP_FEATURES)
    n = len(target_ids)
    seq = np.zeros((n, K, F), dtype=np.float32)
    mask = np.zeros((n, K), dtype=np.float32)
    lengths = np.zeros(n, dtype=int)

    for a, tid in enumerate(target_ids):
        k = ent_of[tid]
        chain = by_ent[k]
        pos = int(np.searchsorted(chain, tid))
        hist = chain[max(0, pos - (K - 1)):pos + 1]
        lengths[a] = len(hist)
        for j, h in enumerate(hist):
            slot = K - len(hist) + j
            r = roll.loc[h] if h in roll.index else None
            feat = [
                float(r["amt_z_entity"]) if r is not None else 0.0,
                float(r["amt_ratio_median"]) if r is not None else 1.0,
                float(r["dlog_time"]) if r is not None else 12.0,
                float(r["device_changed"]) if r is not None else 0.0,
                float(r["addr_changed"]) if r is not None else 0.0,
                float(email_risk.get(h, 0.0)),
                float(r["burst_24h"]) if r is not None else 0.0,
                float(p0_all.get(h, 0.02)),
            ]
            seq[a, slot] = feat
            mask[a, slot] = 1.0
    return seq, mask, lengths


class SeqScaler:
    """Per-feature min-max to [0,1] using train quantiles (robust to outliers)."""

    def __init__(self, lo_q=0.01, hi_q=0.99):
        self.lo_q, self.hi_q = lo_q, hi_q

    def fit(self, seq, mask):
        m = mask.astype(bool)
        self.lo = np.zeros(seq.shape[-1])
        self.hi = np.zeros(seq.shape[-1])
        for f in range(seq.shape[-1]):
            vals = seq[..., f][m]
            self.lo[f] = np.quantile(vals, self.lo_q)
            self.hi[f] = np.quantile(vals, self.hi_q)
            if self.hi[f] - self.lo[f] < 1e-9:
                self.hi[f] = self.lo[f] + 1.0
        return self

    def transform(self, seq):
        z = (seq - self.lo) / (self.hi - self.lo)
        return np.clip(z, 0.0, 1.0).astype(np.float32)
