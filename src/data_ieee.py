"""
IEEE-CIS Fraud Detection loader (real data).

Emits the SAME column schema as ``src/data.py:generate()`` so every downstream
stage (features / sequence / qrc / routing / fusion / distill / evaluate) runs
unchanged, plus a block of raw IEEE engineered columns (C/D/M/V/id/dist/...)
carried alongside for the GBDT baselines -- the HSBC brief explicitly expects
"up to 393 features" for the classical comparison.

Only ``train_transaction.csv`` carries labels (``test_*`` is the unlabelled
Kaggle leaderboard split), so the experiment's own chronological 60/20/20 split
of the training file is the evaluation protocol -- identical to the synthetic
run, which is what makes the two runs directly comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# columns the rest of the pipeline expects (mirrors data.generate())
CANON = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "addr1", "P_emaildomain", "DeviceType", "DeviceInfo",
]
_DERIVED = ["email_risk", "hour", "dow", "fraud_type", "entity"]
# raw categoricals that are not in CANON -> factorized to int codes for the GBDTs
_CAT_EXTRA = ["addr2", "card4", "card5", "card6", "R_emaildomain"]


def _find_dir(cfg: dict) -> Path:
    d = cfg["data"].get("ieee_dir")
    if d and Path(d).expanduser().exists():
        return Path(d).expanduser()
    root = Path(__file__).resolve().parent.parent
    for cand in sorted(root.glob("dataset/*IEEE-CIS*")):
        if cand.is_dir() and (cand / "train_transaction.csv").exists():
            return cand
    raise FileNotFoundError(
        "IEEE-CIS csv directory not found. Set data.ieee_dir in config.yaml or "
        "place the extracted files under experiment/dataset/'... IEEE-CIS ...'/"
    )


def load(cfg: dict, seed: int) -> pd.DataFrame:
    d = cfg["data"]
    ddir = _find_dir(cfg)

    tx = pd.read_csv(ddir / "train_transaction.csv", low_memory=False)
    idf = pd.read_csv(ddir / "train_identity.csv", low_memory=False)
    df = tx.merge(idf, how="left", on="TransactionID")
    del tx, idf

    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # optional systematic, time-span-preserving subsample -------------------
    sub = d.get("ieee_subsample")
    if sub and int(sub) < len(df):
        step = len(df) / float(int(sub))
        keep = np.unique((np.arange(int(sub)) * step).astype(int))
        keep = keep[keep < len(df)]
        df = df.iloc[keep].reset_index(drop=True)

    df = df.copy()  # de-fragment after the merge/slice before many column writes

    # canonical categoricals ---------------------------------------------------
    for c in ("ProductCD", "P_emaildomain", "DeviceType", "DeviceInfo"):
        df[c] = df[c].astype("object")
        df[c] = df[c].where(df[c].notna(), "unknown").astype(str)
    for c in ("card1", "card2", "card3", "addr1"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int64)

    # email_risk: target-encode P_emaildomain on the first 60% (= train split)
    n_tr = max(1, int(cfg["split"]["train"] * len(df)))
    tr = df.iloc[:n_tr]
    glob = float(tr["isFraud"].mean())
    er = tr.groupby("P_emaildomain")["isFraud"].mean()
    df["email_risk"] = df["P_emaildomain"].map(er).fillna(glob).astype(float)

    # derived time / bookkeeping fields --------------------------------------
    df["hour"] = ((df["TransactionDT"] / 3600) % 24).astype(int)
    df["dow"] = ((df["TransactionDT"] / 86400) % 7).astype(int)
    df["fraud_type"] = np.where(df["isFraud"].to_numpy() == 1, "unknown", "none")
    df["entity"] = pd.factorize(
        df["card1"].astype(str) + "_" + df["addr1"].astype(str))[0]

    # raw IEEE feature block for the GBDT baselines --------------------------
    extra: list[str] = []
    reserved = set(CANON) | set(_DERIVED)
    for c in list(df.columns):
        if c in reserved:
            continue
        if c in _CAT_EXTRA:
            df[c] = pd.factorize(df[c].astype(str))[0].astype(np.int32)
        elif c.startswith("M") and c[1:].isdigit():          # M1..M9  (T/F/NaN)
            df[c] = df[c].map({"T": 1.0, "F": 0.0}).astype(np.float32)
        elif df[c].dtype == object:
            df[c] = pd.factorize(df[c].astype(str))[0].astype(np.int32)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        extra.append(c)

    cfg["data"]["_extra_cols"] = extra
    keep = CANON + _DERIVED + extra
    out = df[keep].copy()
    del df
    return out
