# `experiment/` — Abstraction PoC (Hybrid LightGBM + Quantum Reservoir Computing)

Runnable proof-of-concept for the HSBC *"Quantum-Enhanced Credit Card Fraud
Detection"* challenge. It implements the architecture from `../Abstraction.md`
with the fixes from the architecture review, runs the full experiment matrix,
and writes the challenge-required outputs.

```
experiment/
├── config.yaml            all knobs; frozen thresholds written to <outdir>/manifest.json
├── requirements.txt
├── run_experiment.py      thin CLI  ->  python run_experiment.py [--quick] [--data ieee]
├── src/
│   ├── pipeline.py        orchestrator — runs Stage A→H in order, writes <outdir>/
│   ├── data.py            Stage A — synthetic IEEE-CIS-like CNP generator (no Kaggle creds here)
│   ├── data_ieee.py       Stage A — real IEEE-CIS Kaggle csv loader (same schema)
│   ├── features.py        Stage A — static + past-only rolling features (leakage-safe)
│   ├── baselines.py       Stage A* — Logistic Regression, LightGBM, CatBoost (+ OOF scoring)
│   ├── calibration.py     Stage A* — isotonic / Platt + threshold search + Brier/ECE
│   ├── postprocess.py     Stage H — entity-level score smoothing (causal / batch)
│   ├── routing.py         Stage B — ambiguity "review-zone" router
│   ├── sequence.py        Stage C — per-entity sequence builder + angle scaler
│   ├── qrc.py             Stage D — transverse-field Ising quantum reservoir
│   │                                (exact NumPy density-matrix simulator)
│   ├── qrc_pennylane.py   gate-level Braket-portable circuit + cross-check (PennyLane)
│   ├── qrc_braket.py      same circuit, run on the Amazon Braket SDK's own LocalSimulator
│   ├── esn.py             Stage D — classical Echo State Network (control reservoir)
│   ├── fusion.py          Stage E — logistic readout / fusion + recalibration
│   ├── distill.py         Stage F — real-time classical student
│   ├── evaluate.py        metrics + paired bootstrap
│   ├── attribution.py     SHAP / grouped fusion attribution
│   ├── plots.py           diagnostic plots
│   └── report.py          renders <outdir>/report.md from the metrics dicts
└── experiments/           ← generated, one self-contained folder per run
    ├── ieee_full/             PRIMARY: real IEEE-CIS, full 590,540 rows
    │   ├── manifest.json          frozen config, thresholds, seeds, versions, quantum resource
    │   ├── metrics.json           every model × {full test, ambiguous band} + bootstrap deltas
    │   ├── success_criteria.json  pass/fail of the pre-registered criteria
    │   ├── predictions.csv        TransactionID, fraud_probability, fraud_prediction, route, feature_attribution
    │   ├── report.md              auto-generated write-up
    │   ├── run_full.log           run log
    │   ├── plots/*.png
    │   └── code/                  snapshot of src/, run_experiment.py, config.yaml as they
    │                               were when this run was produced (see code/SNAPSHOT_NOTE.md) —
    │                               makes the folder runnable standalone on another machine
    ├── ieee_smoke/            IEEE-CIS, --quick smoke run (same layout as above)
    └── synthetic/             synthetic design-benchmark run (same layout as above)
```

Each `experiments/<name>/` folder is fully self-contained (code snapshot +
config-derived outputs + run log) — copy the folder alone to move a run to
another machine. The same pipeline code produced every experiment (no code
forks between them); only config values and `--data` differ, and the exact
effective config for a given run is frozen into that run's `manifest.json`.

## Run

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_experiment.py --data ieee                      # PRIMARY: real IEEE-CIS (~3.5 h) -> experiments/ieee_full/
./.venv/bin/python run_experiment.py --data synthetic --outdir experiments/synthetic  # design benchmark (~15 min)
./.venv/bin/python run_experiment.py --data ieee --quick --outdir experiments/ieee_smoke  # IEEE smoke run (~8 min)
./.venv/bin/python -m src.qrc_pennylane                               # verify the QRC simulator vs PennyLane
./.venv/bin/python -m src.qrc_braket                                  # verify the QRC simulator vs Amazon Braket
```

Place the extracted Kaggle files under `dataset/…IEEE-CIS…/` (or set
`data.ieee_dir` in `config.yaml`); `--data ieee` auto-detects them.

## What was fixed vs the original `Abstraction.md`

| # | Review finding | Fix in this code |
|---|---|---|
| 1 | `entity_key` is a fragile proxy | `config.data.entity_keys` (E1/E2); rolling features & sequences keyed by it; sensitivity hook |
| 2 | In-sample LightGBM scores break routing | out-of-fold (`cross_val_predict`) train scores so train/val/test score distributions match |
| 3 | Band centred on 0.5 is wrong on imbalanced data | **review-zone** band `[t(recall=0.90), t(recall=0.45)]` in log-odds, capped by routing budget |
| 4 | Fusion output on a different scale wrecks the spliced ranking | post-fit **isotonic recalibration** of the fusion output to band prevalence |
| 5 | "is it just a bigger model?" | ESN control reservoir sized to the QRC embedding dim; `feature_fusion_no_p0` ablation; random-unitary reservoir ablation |
| 6 | Thin expected gains, no "quantum advantage" | pre-registered success criteria = *measurable improvement*; verdict text forbids the phrase |
| 7 | Shot noise on a narrow band | explicit `shots_sweep`; exact bitstring sampling from `diag(rho)` |
| 8 | Barren plateaus | reservoir never trained; only a convex logistic readout (closed-form-ish) |
| 9 | Encoding richness | angle encoding + **data re-uploading** (3 input qubits × 3 rounds ≥ 8 step features) |
| 10 | Spatial multiplexing (Nakajima 2018) | `qrc.spatial_multiplex` = M independent reservoirs, embeddings concatenated |
| 11 | Latency | Stage F distillation + measured `ms/txn` for the real-time student |
| 12 | A fusion head can rank *worse* than `p0` alone once L2 shrinkage spreads weight over noisy reservoir dims | `FusionModel` searches `C` and `class_weight` and, on the **validation** band only, falls back to deploying `p0` unchanged if no fusion head beats it there — a production-safety guardrail, never applied using test data |
| 13 | Splicing a differently-scaled band score into `p0` corrupts the full-population ranking | rank-preserving splice: the band keeps its exact `p0` score interval; only the *order* of rows inside it is replaced by the fusion head, so full-test AUPRC can only move via a genuinely better within-band ranking |
| 14 | Is "more temporal features" alone the whole story? | B4 (standard aggregates) vs **B5** (B4 + hand-built trend/gap features) isolates how much a human analyst already captures by hand; Q3 is compared against both |
| 15 | Does B4's strength depend on picking LightGBM? | Added **CatBoost** (same features as B4) - per the IEEE-CIS 1st-place writeup it was the strongest single GBDT of the three they ensembled, not just an ensemble filler |
| 16 | The 1st-place solution's other big lever besides feature engineering | Added **entity-level score smoothing** (`src/postprocess.py`): a real-time-safe `causal_smooth` (past-only entity mean) and the exact Kaggle `batch_smooth` trick, clearly labelled as audit-only since it uses each entity's *future* transactions and cannot run in HSBC's <300ms path |

## Notes / honesty

- **Primary run = real IEEE-CIS** (`run_experiment.py --data ieee`, `src/data_ieee.py`,
  590,540 transactions) — outputs in `experiments/ieee_full/`. See
  `../Phase2-PoC-Report.md` and `primary datasets compare.md` for the numbers
  and the synthetic-vs-real comparison.
- **Synthetic generator** (`src/data.py`, default `--data synthetic`) is a
  *design benchmark*: it deliberately injects account-takeover / ordered-sequence
  fraud so a temporal model has something to find. Its absolute numbers, and any
  result that depends on injected sequential fraud (B5 gain, shuffle penalty,
  batch-smoothing gain), do **not** carry to real data — every one of them
  weakened or reversed on IEEE-CIS. The original synthetic full run's outputs
  were overwritten by an `ieee_full` run that reused the same output dir before
  this reorg (its log survives as `experiments/ieee_full/synthetic_run_stale.log`,
  its numbers are preserved in `primary datasets compare.md`). Regenerate with
  `run_experiment.py --data synthetic --outdir experiments/synthetic`.
- A 6–8 qubit reservoir is, per Fujii & Nakajima (2016), in the class of a
  ~100–500 node ESN. This PoC is a NISQ-era feature-discovery study, validated
  offline; it never claims quantum advantage.
# Quantum-Reservoir_Fraud-Detection
