# Primary datasets — compare

**Abstraction PoC — Hybrid LightGBM + Quantum Reservoir Computing**
The pipeline run on the **real IEEE-CIS Fraud Detection** dataset (the HSBC
brief's *primary* dataset) vs the synthetic CNP generator (a design benchmark).

| run | rows | config | status | dir |
|---|---|---|---|---|
| **IEEE-CIS (full)** | **590,540** | full, seeds 7 & 13 | ✅ complete (~3.7 h) | `results/` |
| IEEE-CIS (smoke) | 40,000 | `--quick`, 1 seed | ✅ complete | `results_ieee_smoke/` |
| Synthetic (design benchmark) | 77,258 | full, 2 seeds | ✅ complete | `results_synthetic/` * |

\* regenerate with `run_experiment.py --data synthetic --outdir results_synthetic`
(the synthetic full run's outputs were overwritten by the IEEE run; its numbers
are preserved in this document).

**Reading order:** the **IEEE-590k** column is the headline. The synthetic
column is the upper-bound sanity check — the generator *injects*
account-takeover / ordered-sequence fraud on purpose, so it is where the
temporal model has the best possible shot. IEEE-40k is shown only where the
smoke → full transition changes a conclusion (C3).

---

## 0. What was built to run on IEEE-CIS

- **`src/data_ieee.py`** — reads `train_transaction.csv` ⨝ `train_identity.csv`,
  maps to the pipeline's canonical schema, derives `email_risk` (target-encoded
  `P_emaildomain` on the first 60 % chronologically), carries the **422 raw
  IEEE columns** (`C*/D*/M*/V*/id_*/dist*`) for the GBDT baselines — per the
  HSBC brief ("up to 393 features … feature selection expected for the quantum
  step").
- **`run_experiment.py --data ieee`** — everything downstream unchanged
  (`src/pipeline.py` runs Stage A → H identically).
- `test_*.csv` has **no labels** (Kaggle leaderboard split) → evaluation uses
  the same **chronological 60/20/20 split of the training file** as the
  synthetic run. That identical protocol is what makes the columns comparable.

---

## 1. Dataset properties

| | IEEE-CIS (real) | Synthetic |
|---|---|---|
| transactions | **590,540** (train 354,324 / val 118,108 / test 118,108) | 77,258 |
| fraud rate | **3.499 %** (brief: 3.5 %) | 3.63 % |
| entities (E2 = card1+card2+card3+addr1) | 39,974 | 6,000 |
| fraud mechanism | unlabelled | injected: static / sequence-ATO / hard (known) |
| GBDT feature count | ~18 + 422 raw | ~18 |

Real IEEE-CIS fraud is mostly point-in-time detectable, and the "entity" is
only a proxy (no true customer ID), so per-entity history is noisier. → real
data is a harder problem *and* a thinner niche for the QRC.

---

## 2. Full held-out test — AUC-ROC (primary) / AUPRC

| model | IEEE-590k AUROC / AUPRC | Synthetic AUROC / AUPRC |
|---|---|---|
| B0 logreg (static) | 0.718 / 0.113 | 0.833 / 0.516 |
| B1 LightGBM (static) | 0.773 / 0.170 | 0.803 / 0.533 |
| **B4 LightGBM (+raw feats)** | **0.884 / 0.478** | **0.912 / 0.688** |
| B4 calibrated (= `p0`) | 0.884 / 0.464 | 0.912 / 0.674 |
| B5 LightGBM (+ manual temporal FE) | 0.883 / 0.480 | 0.918 / 0.719 |
| B_CatBoost | 0.885 / 0.453 | 0.909 / 0.687 |
| ESN hybrid | 0.883 / 0.464 | 0.913 / 0.670 |
| **Q3 QRC hybrid (deployed)** | 0.883 / 0.463 | 0.912 / 0.674 |

**HSBC reference** (challenge §4.1): IEEE-CIS Kaggle 1st place **AUC-ROC 0.9459**
(3-GBDT ensemble). Our B4 ≈ 0.884 is lower by design: no hand-built
UID/aggregation "magic" features, a genuine chronological holdout (harder than
Kaggle's split), no ensembling. It is an honest tuned-single-GBDT floor, which
is the correct baseline for a "does quantum beat it?" question.

---

## 3. Pre-registered success criteria

| # | criterion | IEEE-590k | Synthetic |
|---|---|---|---|
| C1 | ΔAUPRC(Q3 − B4) full test, CI > 0 | **fail** −0.0151 [−0.0165, −0.0137] | **fail** −0.0142 [−0.0168, −0.0117] |
| C2 | false-decline ↓ at recall 0.90, CI < 0 | **fail** +0.0194 [+0.0034, +0.0415] | **fail** +0.0051 [+0.0046, +0.126] |
| C3 | Q2 ≥ parity with classical ESN on band (CI low > −0.02) | **PASS (parity)** −0.0039 [−0.0071, −0.0007] | **PASS (parity)** +0.0191 [−0.0099, +0.0486] |
| C4 | holds across reservoir seeds | fail | fail |
| **Overall measurable improvement** | **NO** | **NO** |

On real data the QRC hybrid is **further behind** the tuned GBDT (C1: −0.0151 vs
−0.0142). C3 still passes the −0.02 parity bar, but the sign flips: on synthetic
the hybrid was *above* the ESN (+0.019), on IEEE it is *below* (−0.004) — parity,
never a win. (On the IEEE-40k smoke run C3 *failed*: band CI low −0.026 < −0.02.
The full 590k run, with 1,556 band positives instead of 75, tightens the interval
enough to pass parity — the conclusion "does not beat the ESN" is unchanged.)

---

## 4. Ambiguous band — where the QRC actually operates

| | IEEE-590k | Synthetic |
|---|---|---|
| routing rate (test) | 17.0 % | 12.9 % |
| fraud in band vs overall | 7.7 % vs 3.4 % (2.2×) | 7.3 % vs 3.1 % (2.3×) |
| band test positives | **1,556** | 145 |
| B4 `p0` band AUPRC | **0.1344** | 0.189 |
| Q0 QRC-raw band AUPRC | 0.1315 | 0.150 |
| Q1 score-fusion band AUPRC (deployed) | 0.1311 | 0.189 |
| Q2 feature-fusion band AUPRC | 0.1327 | 0.142 |
| ESN qrc_only band AUPRC | 0.1361 | — |
| ESN score-fusion band AUPRC (deployed) | 0.1350 | — |
| ESN feature-fusion band AUPRC | 0.1365 | — |

On the 590k run **every ESN head ≥ its QRC counterpart** on the band
(0.1350–0.1365 vs 0.1311–0.1327), and QRC-raw (Q0 = 0.1315) is *below*
LightGBM's own `p0` (0.1344). This is the C3 "is the quantum reservoir doing
anything a classical one can't?" question, and on real data at scale the answer
is **no**.

The deployment guard fired on **both** datasets — no fusion head beat `p0` on
the validation band, so the system safely deployed `p0` unchanged. Deployed head
= `score_fusion` at `C=0.01` (maximum regularisation → essentially `p0`). The
hybrid never hurts production; it also never helps.

---

## 5. Order-sensitivity — the QRC's one distinctive property

Band AUPRC, ablations, **guard disabled** (Δ vs the unguarded "ideal" head):

| ablation | IEEE-590k Δ | Synthetic Δ | reading |
|---|---|---|---|
| ideal unguarded head | 0.1327 (ref) | 0.1423 (ref) | — |
| **shuffle sequence** | **+0.0011** | **−0.0085** | IEEE: shuffling *helps* → no ordered signal used. Synthetic: order helps |
| no history (K=1) | +0.0039 | +0.0114 | trajectory barely matters on both; slightly *harmful* on both |
| random reservoir | +0.0012 | +0.0279 | a random unitary does ≥ as well — Ising tuning isn't earning its keep |
| no entanglement (J=0) | +0.0019 | −0.0011 | the XX coupling is nearly inert at current J,h |
| shots 128 / 1024 / 8192 (Δ vs ideal) | −0.0369 / −0.0155 / −0.0013 | −0.046 / −0.029 / −0.007 | smooth monotone; on IEEE ~8192 shots ≈ recovers ideal |

**This is the key negative result.** On synthetic data the reservoir provably
used temporal order (shuffling hurt). On real IEEE-CIS band transactions,
shuffling the sequence *improves* the score — the reservoir is not extracting
useful ordered-sequence structure, because real IEEE fraud in the band is not
the temporal-sequence phenomenon the synthetic generator injected.

Multi-seed (reservoir seeds 7, 13): C1 = −0.0151 / −0.0145, C3 vs ESN = −0.0039
/ −0.0032 — identical conclusions across seeds; the outcome is not a seed
artifact.

---

## 6. What transferred — the classical levers

| lever | IEEE-590k ΔAUPRC | Synthetic ΔAUPRC | verdict |
|---|---|---|---|
| **causal entity smoothing** (real-time-safe, past-only card mean) | **+0.0071 [+0.0027, +0.0112]** | +0.0010 [−0.002, +0.004] ns | **significant win on real data** |
| batch entity smoothing (Kaggle trick, non-deployable) | **−0.2113 [−0.224, −0.198]** | +0.0494 [+0.031, +0.069] | huge reversal — was leaderboard overfit; proxy entity key mixes fraud/legit |
| B5 vs B4 — manual temporal FE (amount trend, gap ratio) | +0.0017 [−0.0007, +0.0043] ns | +0.0305 [+0.018, +0.045] | real & significant on synthetic only; ns on the full IEEE run |
| CatBoost vs B4 | −0.0245 [−0.030, −0.019] | −0.0016 [−0.012, +0.010] | B4's strength is not a LightGBM artifact |
| router concentrates fraud | 7.7 % vs 3.4 % | 7.3 % vs 3.1 % | works on both |
| distilled student latency | 0.0031 ms/txn | 0.0034 ms/txn | fits the <300 ms budget with ~5 orders of magnitude to spare on both |

---

## 7. Reading

1. **The pipeline transfers cleanly** to real IEEE-CIS — the loader + schema
   mapping was the only work; every downstream stage ran unchanged. Absolute
   metrics drop (harder problem, proxy entity key) exactly as expected.
2. **The pre-registered conclusion is unchanged and stronger on real data:**
   the quantum-reservoir hybrid does **not** beat the tuned GBDT (C1 worsens
   −0.0142 → −0.0151), and on the band it moves from *above* the size-matched
   ESN (+0.019 synthetic) to *below* it (−0.004 IEEE) — parity at best, never a
   win.
3. **The reservoir's one distinctive behaviour — order-sensitivity — does not
   appear on real IEEE-CIS.** Shuffling the input sequence *improves* the band
   score. The architecture's central thesis (temporal second opinion on
   ambiguous transactions) does not find purchase on this dataset.
4. **Every positive result is classical:** causal entity smoothing (now
   statistically significant), isotonic calibration, and the ambiguity router.
   Manual temporal FE (B5) and the batch-smoothing "magic" both evaporate on
   real data (+0.031 → +0.002 ns; +0.049 → −0.211) — they were synthetic-
   injection / Kaggle-leaderboard artefacts, not signal.
5. **For the HSBC submission:** lead with the deployable classical stack —
   router + causal entity smoothing + distillation — as the measurable
   contribution. Present the QRC honestly as a NISQ-era feature-discovery study
   with a **documented null result** and two concrete, ablation-backed next
   steps: (a) re-tune the Ising Hamiltonian toward the random-reservoir
   result, (b) redesign the trajectory encoding (single-step / attention-pooled
   instead of mean-delta). Never claim quantum advantage — the pre-registered
   verdict is the only verdict.
