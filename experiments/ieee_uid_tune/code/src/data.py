"""
Synthetic card-not-present transaction generator.

Why synthetic: the IEEE-CIS dataset requires Kaggle credentials + competition
rules acceptance, which are not available in this environment.  The generator
is built to reproduce the properties HSBC's challenge statement calls out so
that the *pipeline* and the *QRC contribution* can be evaluated honestly:

  * ~3.5 % fraud rate, severe class imbalance
  * card-not-present only (online, card absent)
  * per-entity (card) transaction history with realistic inter-arrival times
  * TWO fraud regimes:
      - "static"   : obvious from a single transaction (big amount, new device,
                     address mismatch, risky email)  -> a good GBDT catches these
      - "sequence" : account-takeover.  After a dormant period a burst of
                     transactions with gradually escalating amounts and a
                     persistent device change.  Each single transaction is only
                     mildly unusual (lands in the ambiguous band); the *sequence*
                     is the giveaway  -> temporal models (QRC / ESN / LSTM) help
      - "hard"     : almost no signal anywhere  -> mostly missed (realistic)
  * legitimate-but-suspicious transactions (holiday spend, travel, new phone)
    that a static model wrongly flags -> false-decline candidates the temporal
    context should rescue.

Feature names mirror IEEE-CIS where possible (TransactionDT, TransactionAmt,
ProductCD, card1-3, addr1, P_emaildomain, DeviceType, DeviceInfo, ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = ["W", "C", "R", "H", "S"]
DEVICE_TYPES = ["desktop", "mobile"]
DEVICE_MODELS = [f"dev_{i:02d}" for i in range(40)]
EMAIL_DOMAINS = [f"mail_{i:02d}" for i in range(25)]


def _poisson_times(rate_per_day: float, days: int, rng: np.random.Generator) -> np.ndarray:
    """Homogeneous Poisson arrival times (seconds) over `days`."""
    total = days * 86400
    n = rng.poisson(rate_per_day * days)
    if n == 0:
        n = 1
    t = np.sort(rng.uniform(0, total, size=n))
    return t


def generate(cfg: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = cfg["data"]
    n_entities = d["n_entities"]
    days = d["days"]

    # ---- entity profiles ------------------------------------------------
    ent_base_amt = rng.lognormal(mean=3.6, sigma=0.6, size=n_entities)          # ~ $37 typical
    ent_home_addr = rng.integers(100, 999, size=n_entities)
    ent_device_type = rng.integers(0, 2, size=n_entities)
    ent_device_model = rng.integers(0, len(DEVICE_MODELS), size=n_entities)
    ent_email = rng.integers(0, len(EMAIL_DOMAINS), size=n_entities)
    ent_email_risk = np.clip(rng.beta(1.5, 8, size=n_entities), 0, 1)           # mostly low
    # transactions per DAY, mean = avg_txn_per_entity / days
    ent_rate = np.clip(
        rng.gamma(2.0, (d["avg_txn_per_entity"] / days) / 2.0, size=n_entities),
        1e-4, None,
    )
    ent_card1 = rng.integers(1000, 9999, size=n_entities)
    ent_card2 = rng.integers(100, 999, size=n_entities)
    ent_card3 = rng.integers(100, 999, size=n_entities)

    rows = []
    for e in range(n_entities):
        times = _poisson_times(ent_rate[e], days, rng)
        # normal behaviour ------------------------------------------------
        n = len(times)
        amt = ent_base_amt[e] * rng.lognormal(0.0, 0.35, size=n)
        dev_type = np.full(n, ent_device_type[e])
        dev_model = np.full(n, ent_device_model[e])
        addr = np.full(n, ent_home_addr[e])
        email_risk = np.full(n, ent_email_risk[e]) + rng.normal(0, 0.02, n)
        product = rng.integers(0, len(PRODUCTS), size=n)
        is_fraud = np.zeros(n, dtype=int)
        ftype = np.array(["none"] * n, dtype=object)

        # occasional legit anomalies -----------------------------------
        # Designed to OVERLAP with sequence fraud on per-transaction and
        # simple aggregate features, so that the *only* clean separator is
        # the ordered temporal shape (escalation vs flat, dormancy-then-burst
        # vs steady).
        for i in range(n):
            if rng.random() < d["frac_legit_anomalous"]:
                kind = rng.integers(0, 4)
                if kind == 0:                      # big legitimate purchase
                    amt[i] *= rng.uniform(4, 9)
                elif kind == 1:                    # travel
                    addr[i] = rng.integers(100, 999)
                elif kind == 2:                    # bought a new device
                    newdev = rng.integers(0, len(DEVICE_MODELS))
                    dev_model[i:] = newdev
                    if rng.random() < 0.5:
                        dev_type[i:] = rng.integers(0, 2)
                elif n - i >= 3:                   # shopping spree: a FLAT burst
                    span = int(rng.integers(2, min(5, n - i)))
                    burst_amt = ent_base_amt[e] * rng.uniform(1.4, 2.8)
                    for j in range(span):
                        if i + j < n:
                            amt[i + j] = burst_amt * rng.lognormal(0, 0.12)  # no trend
                    t_end = min(i + span, n)
                    times[i:t_end] = np.sort(
                        times[i] + rng.uniform(0, rng.uniform(0.4, 2.0) * 86400,
                                               size=t_end - i))
                    if rng.random() < 0.3:         # spree on a new tablet
                        dev_model[i:t_end] = rng.integers(0, len(DEVICE_MODELS))
                else:
                    continue
                ftype[i] = "legit_anomalous"

        # sequential (account-takeover) fraud --------------------------
        # The tell is the *sequence*: a dormant gap, then a compressed burst
        # with a monotonic amount escalation.  Each single transaction is only
        # mildly unusual; no static "risky email" tell; device change only
        # ~55% of the time (session hijack keeps the saved device).
        if n >= 5 and rng.random() < d.get("p_sequence_fraud_per_entity", 0.06):
            start = int(rng.integers(2, n - 2))
            burst_len = int(min(rng.integers(3, 6), n - start))
            gap = rng.uniform(6, 18) * 86400
            times[start:] = np.minimum(times[start:] + gap, days * 86400 - 1)
            burst_span = rng.uniform(0.3, 2.0) * 86400
            times[start:start + burst_len] = np.sort(
                times[start] + rng.uniform(0, burst_span, size=burst_len)
            )
            change_dev = rng.random() < 0.55
            new_dev = int(rng.integers(0, len(DEVICE_MODELS)))
            lo_e, hi_e = rng.uniform(1.05, 1.25), rng.uniform(1.9, 2.8)
            esc = np.linspace(lo_e, hi_e, burst_len)
            for j in range(burst_len):
                idx = start + j
                amt[idx] = ent_base_amt[e] * esc[j] * rng.lognormal(0, 0.10)
                if change_dev:
                    dev_model[idx] = new_dev
                is_fraud[idx] = 1
                ftype[idx] = "sequence_fraud"

        # static (single-shot) fraud ----------------------------------
        for i in range(n):
            if is_fraud[i] == 0 and rng.random() < d.get("p_static_fraud_per_txn", 0.016):
                amt[i] = ent_base_amt[e] * rng.uniform(5, 18)
                dev_model[i] = rng.integers(0, len(DEVICE_MODELS))
                dev_type[i] = rng.integers(0, 2)
                addr[i] = rng.integers(100, 999)
                email_risk[i] = rng.uniform(0.55, 0.95)
                is_fraud[i] = 1
                ftype[i] = "static_fraud"
            elif is_fraud[i] == 0 and rng.random() < d.get("p_hard_fraud_per_txn", 0.006):
                amt[i] *= rng.uniform(1.3, 2.2)
                email_risk[i] += 0.05
                is_fraud[i] = 1
                ftype[i] = "hard_fraud"

        for i in range(n):
            rows.append(
                dict(
                    entity=e,
                    TransactionDT=float(times[i]),
                    TransactionAmt=float(max(amt[i], 1.0)),
                    ProductCD=PRODUCTS[product[i]],
                    card1=int(ent_card1[e]),
                    card2=int(ent_card2[e]),
                    card3=int(ent_card3[e]),
                    addr1=int(addr[i]),
                    P_emaildomain=EMAIL_DOMAINS[ent_email[e]],
                    email_risk=float(np.clip(email_risk[i], 0, 1)),
                    DeviceType=DEVICE_TYPES[int(dev_type[i]) % 2],
                    DeviceInfo=DEVICE_MODELS[int(dev_model[i]) % len(DEVICE_MODELS)],
                    isFraud=int(is_fraud[i]),
                    fraud_type=ftype[i],
                )
            )

    df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
    df.insert(0, "TransactionID", np.arange(len(df)))

    # rebalance fraud rate to the target by dropping surplus fraud rows
    target = d["fraud_rate"]
    cur = df["isFraud"].mean()
    if cur > target:
        fraud_idx = df.index[df["isFraud"] == 1].to_numpy()
        n_keep = int(round(target * len(df) / (1 - target)))
        drop = rng.choice(fraud_idx, size=max(len(fraud_idx) - n_keep, 0), replace=False)
        df = df.drop(index=drop).reset_index(drop=True)
        df["TransactionID"] = np.arange(len(df))

    # derived time fields
    df["hour"] = ((df["TransactionDT"] / 3600) % 24).astype(int)
    df["dow"] = ((df["TransactionDT"] / 86400) % 7).astype(int)
    return df


def chronological_split(df: pd.DataFrame, cfg: dict):
    n = len(df)
    a = int(cfg["split"]["train"] * n)
    b = a + int(cfg["split"]["val"] * n)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def add_entity_key(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    df["entity_key"] = df[list(cols)].astype(str).agg("-".join, axis=1)
    return df
