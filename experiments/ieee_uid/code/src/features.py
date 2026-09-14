"""
Feature engineering with a hard anti-leakage rule:

    every feature for a transaction at time t uses ONLY rows with time < t.

Two feature sets:
  * static   : per-transaction fields (log amount, hour, product, device, ...)
  * rolling  : per-entity running statistics computed strictly from the past
               (this is the "temporal feature engineering" baseline B4 that QRC
                must beat).

All encoders / scalers are fit on TRAIN only and applied to val / test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATIC_NUM = ["log_amt", "hour_sin", "hour_cos", "dow", "email_risk"]
STATIC_CAT = ["ProductCD", "DeviceType"]

# The rolling features a bank would routinely engineer (aggregates + a short
# linear trend). This is the strong-but-standard baseline B4 must be measured
# against; the ordered-sequence models (QRC / ESN / LSTM) additionally see the
# full step-by-step trajectory.
B4_ROLLING = [
    "amt_z_entity", "amt_ratio_median", "dlog_time", "device_changed",
    "addr_changed", "burst_24h", "entity_txn_idx", "entity_seen",
]
# extra hand-built temporal features -> the "aggressive manual FE" upper
# baseline B5, used only to show how much of the gap manual work closes.
B5_EXTRA = ["amt_trend_3", "gap_prev_ratio", "amt_ratio_p95",
            "time_since_last_device_change"]

ROLLING = [
    "amt_z_entity", "amt_ratio_median", "dlog_time", "device_changed",
    "addr_changed", "burst_24h", "entity_txn_idx", "entity_seen",
    "amt_ratio_p95", "time_since_last_device_change",
]


def _freq_encode(train: pd.Series, other: pd.Series) -> pd.Series:
    freq = train.value_counts(normalize=True)
    return other.map(freq).fillna(0.0)


def build_static(train, val, test):
    out = []
    for df in (train, val, test):
        f = pd.DataFrame(index=df.index)
        f["log_amt"] = np.log1p(df["TransactionAmt"].to_numpy())
        f["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        f["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        f["dow"] = df["dow"].to_numpy()
        f["email_risk"] = df["email_risk"].to_numpy()
        out.append(f)
    ftr, fva, fte = out
    for c in STATIC_CAT:
        ftr[c + "_freq"] = _freq_encode(train[c], train[c])
        fva[c + "_freq"] = _freq_encode(train[c], val[c])
        fte[c + "_freq"] = _freq_encode(train[c], test[c])
    for c in ["card1", "addr1", "P_emaildomain", "DeviceInfo"]:
        ftr[c + "_freq"] = _freq_encode(train[c], train[c])
        fva[c + "_freq"] = _freq_encode(train[c], val[c])
        fte[c + "_freq"] = _freq_encode(train[c], test[c])
    return ftr, fva, fte


def build_rolling(df_all: pd.DataFrame, entity_col: str = "entity_key") -> pd.DataFrame:
    """Past-only per-entity running features for the whole (time-sorted) frame."""
    df = df_all.sort_values("TransactionDT").reset_index(drop=True)
    f = pd.DataFrame(index=df.index)

    amt = df["TransactionAmt"].to_numpy(dtype=float)
    dt = df["TransactionDT"].to_numpy(dtype=float)
    dev = df["DeviceInfo"].to_numpy()
    addr = df["addr1"].to_numpy()
    keys = df[entity_col].to_numpy()

    amt_z = np.zeros(len(df))
    amt_med = np.zeros(len(df))
    amt_p95 = np.zeros(len(df))
    dlog_t = np.zeros(len(df))
    dev_ch = np.zeros(len(df))
    addr_ch = np.zeros(len(df))
    burst = np.zeros(len(df))
    idx_in_ent = np.zeros(len(df))
    seen = np.zeros(len(df))
    since_dev_ch = np.zeros(len(df))
    amt_trend = np.zeros(len(df))       # OLS slope of last <=3 log-amounts (incl. current)
    gap_prev_ratio = np.zeros(len(df))  # (prev inter-arrival) / (median inter-arrival)

    from collections import defaultdict
    hist_amt = defaultdict(list)
    hist_t = defaultdict(list)
    last_dev = {}
    last_addr = {}
    last_dev_ch_t = {}

    for i in range(len(df)):
        k = keys[i]
        ha, ht = hist_amt[k], hist_t[k]
        seen[i] = len(ha)
        idx_in_ent[i] = len(ha)
        if ha:
            la = np.log1p(np.asarray(ha))
            mu, sd = la.mean(), la.std() + 1e-6
            amt_z[i] = (np.log1p(amt[i]) - mu) / sd
            med = np.median(ha)
            amt_med[i] = amt[i] / (med + 1e-6)
            amt_p95[i] = amt[i] / (np.percentile(ha, 95) + 1e-6)
            dlog_t[i] = np.log1p(dt[i] - ht[-1])
            dev_ch[i] = float(dev[i] != last_dev[k])
            addr_ch[i] = float(addr[i] != last_addr[k])
            burst[i] = int(np.sum(np.asarray(ht) > dt[i] - 86400))
            since_dev_ch[i] = np.log1p(dt[i] - last_dev_ch_t.get(k, dt[i]))
            recent = np.log1p(np.asarray((ha + [amt[i]])[-3:]))
            if len(recent) >= 2:
                xx = np.arange(len(recent))
                amt_trend[i] = np.polyfit(xx, recent, 1)[0]
            if len(ht) >= 2:
                gaps = np.diff(np.asarray(ht))
                med_gap = np.median(gaps) if len(gaps) else 1.0
                gap_prev_ratio[i] = (dt[i] - ht[-1]) / (med_gap + 1.0)
        else:
            amt_z[i] = 0.0
            amt_med[i] = 1.0
            amt_p95[i] = 1.0
            dlog_t[i] = np.log1p(7 * 86400)
            dev_ch[i] = 0.0
            addr_ch[i] = 0.0
            burst[i] = 0
            since_dev_ch[i] = 0.0
            amt_trend[i] = 0.0
            gap_prev_ratio[i] = 1.0

        if k not in last_dev or dev[i] != last_dev[k]:
            last_dev_ch_t[k] = dt[i]
        hist_amt[k].append(amt[i])
        hist_t[k].append(dt[i])
        last_dev[k] = dev[i]
        last_addr[k] = addr[i]

    f["amt_z_entity"] = amt_z
    f["amt_ratio_median"] = np.clip(amt_med, 0, 50)
    f["amt_ratio_p95"] = np.clip(amt_p95, 0, 50)
    f["dlog_time"] = dlog_t
    f["device_changed"] = dev_ch
    f["addr_changed"] = addr_ch
    f["burst_24h"] = burst
    f["entity_txn_idx"] = idx_in_ent
    f["entity_seen"] = np.clip(seen, 0, 100)
    f["time_since_last_device_change"] = since_dev_ch
    f["amt_trend_3"] = np.clip(amt_trend, -5, 5)
    f["gap_prev_ratio"] = np.clip(gap_prev_ratio, 0, 50)
    f["TransactionID"] = df["TransactionID"].to_numpy()
    return f.set_index("TransactionID")


def assemble(static_df, rolling_df, txn_ids, rolling="b4"):
    """rolling: 'none' | 'b4' (standard aggregates) | 'b5' (b4 + manual temporal FE)."""
    s = static_df.copy()
    s.index = txn_ids
    if rolling == "none":
        return s
    cols = B4_ROLLING + (B5_EXTRA if rolling == "b5" else [])
    r = rolling_df.reindex(txn_ids)[cols]
    return pd.concat([s, r], axis=1)
