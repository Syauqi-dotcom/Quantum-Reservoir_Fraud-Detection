# Primary datasets — compare

**Abstraction PoC — Hybrid LightGBM + Quantum Reservoir Computing**
The pipeline run on (a) the synthetic CNP generator vs (b) the **real IEEE-CIS
Fraud Detection** dataset — the HSBC brief's *primary* dataset.

| run | rows | config | status | dir |
|---|---|---|---|---|
| Synthetic | 77,258 | full, 2 seeds | ✅ complete | `results/` |
| IEEE-CIS (smoke) | 40,000 | `--quick`, 1 seed | ✅ complete end-to-end | `results_ieee_smoke/` |
| IEEE-CIS (full) | 590,540 | full, 2 seeds | ⏳ baselines/routing/smoothing done; QRC + ablations running | `results_ieee/` |

Numbers below: **Synthetic** and **IEEE-40k** columns are final. **IEEE-590k**
column filled where the full run has already produced it; `…` = still running.

---

## 0. What was built to run on IEEE-CIS

- **`src/data_ieee.py`** — reads `train_transaction.csv` ⨝ `train_identity.csv`,
  maps to the pipeline's canonical schema, derives `email_risk` (target-encoded
  `P_emaildomain` on the first 60 % chronologically), carries the **422 raw
  IEEE columns** (`C*/D*/M*/V*/id_*/dist*`) for the GBDT baselines — per the
  HSBC brief ("up to 393 features … feature selection expected for the quantum
  step").
- **`run_experiment.py --data ieee [--outdir results_ieee]`** — everything
  downstream unchanged.
- `test_*.csv` has **no labels** (Kaggle leaderboard split) → evaluation uses
  the same **chronological 60/20/20 split of the training file** as the
  synthetic run. That identical protocol is what makes the columns comparable.

---

## 1. Dataset properties

| | Synthetic | IEEE-CIS (real) |
|---|---|---|
| transactions | 77,258 | 590,540 |
| fraud rate | 3.63 % | **3.499 %** (brief: 3.5 %) |
| entities (E2 = card1+card2+card3+addr1) | 6,000 | 39,974 |
| fraud mechanism | injected: static / sequence-ATO / hard (known) | unlabelled |
| GBDT feature count | ~18 | ~18 + 422 raw |

The synthetic generator **injects account-takeover / ordered-sequence fraud on
purpose** so a temporal model has something to find. Real IEEE-CIS fraud is
mostly point-in-time detectable, and the "entity" is only a proxy (no true
customer ID), so per-entity history is noisier. → real data is a harder problem
*and* a thinner niche for the QRC.

---

## 2. Full held-out test — AUC-ROC (primary) / AUPRC

| model | Synthetic AUROC / AUPRC | IEEE-40k AUROC / AUPRC | IEEE-590k |
|---|---|---|---|
| B0 logreg (static) | 0.833 / 0.516 | 0.719 / 0.102 | … |
| B1 LightGBM (static) | 0.803 / 0.533 | 0.717 / 0.111 | … |
| **B4 LightGBM (+raw feats)** | **0.912 / 0.688** | **0.856 / 0.438** | AUPRC 0.464 (cal.) |
| B4 calibrated | 0.912 / 0.674 | 0.854 / 0.403 | 0.464 |
| B5 LightGBM (+ manual temporal FE) | 0.918 / 0.719 | 0.864 / 0.459 | … |
| B_CatBoost | 0.909 / 0.687 | 0.862 / 0.428 | … |
| ESN hybrid | 0.913 / 0.670 | 0.854 / 0.402 | … |
| **Q3 QRC hybrid (deployed)** | 0.912 / 0.674 | 0.854 / 0.403 | … |

**HSBC reference** (challenge §4.1): IEEE-CIS Kaggle 1st place **AUC-ROC 0.9459**
(3-GBDT ensemble). Our B4 ≈ 0.85–0.86 is lower by design: no hand-built
UID/aggregation "magic" features, a genuine chronological holdout (harder than
Kaggle's split), no ensembling. It is an honest tuned-single-GBDT floor, which
is the correct baseline for a "does quantum beat it?" question.

---

## 3. Pre-registered success criteria

| # | criterion | Synthetic | IEEE-40k |
|---|---|---|---|
| C1 | ΔAUPRC(Q3 − B4) full test, CI > 0 | **fail** −0.014 [−0.017, −0.012] | **fail** −0.035 [−0.043, −0.027] |
| C2 | false-decline ↓ at recall 0.90, CI < 0 | **fail** +0.005 [+0.005, +0.126] | **fail** +0.028 [+0.004, +0.158] |
| C3 | Q2 ≥ parity with classical ESN on band (CI low > −0.02) | **PASS** +0.019 [−0.010, +0.049] | **fail** +0.012 [−0.026, +0.046] |
| C4 | holds across reservoir seeds | fail | fail (1 seed in quick) |
| **Overall measurable improvement** | **NO** | **NO** |

On real data the QRC hybrid is **further behind** the tuned GBDT (C1: −0.035 vs
−0.014), and the parity-with-ESN result (C3) **flips to fail** — the band CI low
(−0.026) drops below the −0.02 parity bar.

---

## 4. Ambiguous band — where the QRC actually operates

| | Synthetic | IEEE-40k | IEEE-590k |
|---|---|---|---|
| routing rate (test) | 12.9 % | 20.1 % | 17.1 % |
| fraud in band vs overall | 7.3 % vs 3.1 % (2.3×) | 4.7 % vs 3.3 % (1.4×) | 7.7 % vs 3.4 % (2.2×) |
| band test positives | 145 | 75 | **1,556** |
| B4 p0 band AUPRC | 0.189 | 0.079 | … (metrics computing) |
| Q0 QRC-raw band AUPRC | 0.150 | 0.078 | 0.1315 |
| Q1 score-fusion band AUPRC | 0.189 | 0.079 | 0.1311 |
| Q2 feature-fusion band AUPRC | 0.142 | 0.079 | 0.1327 |
| **ESN qrc_only band AUPRC** | — | 0.078 (deployed 0.067) | **0.1361** |
| **ESN feature_fusion band AUPRC** | — | — | **0.1365** |

On the 590k run the **classical ESN edges the QRC on the band** (0.1361–0.1365
vs 0.1311–0.1327) — every ESN head ≥ its QRC counterpart. This is the C3
"is the quantum reservoir doing anything a classical one can't?" question, and
on real data at scale the answer is **no**. Deployed head = `score_fusion` with
`C=0.01` (maximum regularisation → essentially `p0`); the guard's fallback held.

The deployment guard fired on **both** datasets — no fusion head beat `p0` on
the validation band, so the system safely deployed `p0` unchanged. The hybrid
never hurts production; it also never helps.

---

## 5. Order-sensitivity — the QRC's one distinctive property

Band AUPRC, ablations, **guard disabled** (Δ vs the unguarded "ideal" head):

| ablation | Synthetic Δ | IEEE-40k Δ | reading |
|---|---|---|---|
| ideal unguarded head | 0.1423 (ref) | 0.0580 (ref) | — |
| **shuffle sequence** | **−0.0085** | **+0.0197** | synthetic: order helps. IEEE: shuffling *helps* → no ordered signal used |
| no history (K=1) | +0.0114 | −0.0069 | trajectory barely matters on both |
| random reservoir | +0.0279 | +0.0115 | a random unitary does ≥ as well — Ising tuning isn't earning its keep |
| no entanglement (J=0) | −0.0011 | −0.0034 | the XX coupling is nearly inert at current J,h |

**This is the key negative result.** On synthetic data the reservoir provably
used temporal order (shuffling hurt). On real IEEE-CIS band transactions,
shuffling the sequence *improves* the score — the reservoir is not extracting
useful ordered-sequence structure, because real IEEE fraud in the band is not
the temporal-sequence phenomenon the synthetic generator injected.

---

## 6. What transferred — the classical levers

| lever | Synthetic ΔAUPRC | IEEE ΔAUPRC | verdict |
|---|---|---|---|
| **causal entity smoothing** (real-time-safe, past-only card mean) | +0.0010 [−0.002, +0.004] ns | **+0.0071 [+0.003, +0.011]** (590k) | **significant win on real data** |
| batch entity smoothing (Kaggle trick, non-deployable) | +0.0494 [+0.031, +0.069] | **−0.2113 [−0.224, −0.198]** (590k) | huge reversal — was leaderboard overfit; proxy entity key mixes fraud/legit |
| B5 vs B4 — manual temporal FE (amount trend, gap ratio) | +0.0305 [+0.018, +0.045] | +0.0211 [+0.009, +0.034] (40k) | real positive on both; a human analyst out-captures the reservoir |
| CatBoost vs B4 | −0.0016 [−0.012, +0.010] | −0.0096 [−0.032, +0.015] (40k) | B4's strength is not a LightGBM artifact |
| router concentrates fraud | 7.3 % vs 3.1 % | 7.7 % vs 3.4 % (590k) | works on both |
| distilled student latency | 0.0034 ms/txn | 0.0038 ms/txn | fits the <300 ms budget with ~5 orders of magnitude to spare on both |

---

## 7. Reading

1. **The pipeline transfers cleanly** to real IEEE-CIS — loader + schema mapping
   was the only work; every downstream stage ran unchanged. Absolute metrics
   drop (harder problem, proxy entity key) exactly as expected.
2. **The pre-registered conclusion is unchanged and stronger on real data:**
   the quantum-reservoir hybrid does **not** beat the tuned GBDT (C1 worsens
   −0.014 → −0.035), and now does **not** even reach parity with a
   size-matched classical ESN on the band (C3 flips PASS → fail).
3. **The reservoir's one distinctive behaviour — order-sensitivity — does not
   appear on real IEEE-CIS.** Shuffling the input sequence *improves* the band
   score. The architecture's central thesis (temporal second opinion on
   ambiguous transactions) does not find purchase on this dataset.
4. **Every positive result is classical:** causal entity smoothing (now
   statistically significant), manual temporal feature engineering, isotonic
   calibration, and the ambiguity router. The batch-smoothing "magic" evaporates
   (+0.05 → −0.21) — it was Kaggle-leaderboard overfitting, not signal.
5. **For the HSBC submission:** lead with the deployable classical stack —
   router + causal entity smoothing + distillation — as the measurable
   contribution. Present the QRC honestly as a NISQ-era feature-discovery study
   with a **documented null result** and two concrete, ablation-backed next
   steps: (a) re-tune the Ising Hamiltonian toward the random-reservoir
   result, (b) redesign the trajectory encoding (single-step / attention-pooled
   instead of mean-delta). Never claim quantum advantage — the pre-registered
   verdict is the only verdict.
