#!/usr/bin/env python3
"""
Abstraction - end-to-end PoC runner (thin CLI around ``src/pipeline.py``).

    python run_experiment.py --data ieee                             # PRIMARY: real IEEE-CIS Kaggle csv files
    python run_experiment.py --data synthetic --outdir experiments/synthetic  # design benchmark
    python run_experiment.py --data ieee --quick                     # fast smoke run

Each run writes a self-contained folder (default under experiments/,
override with --outdir) so a single run can be copied whole to another
machine:
    manifest.json          frozen config, thresholds, seeds, versions, timings
    metrics.json           every model's AUC-ROC / AUPRC / F1 / precision / recall
                           / confusion matrix, on full test + ambiguous band,
                           plus paired-bootstrap deltas
    success_criteria.json  pass / fail of the pre-registered criteria
    predictions.csv        TransactionID, fraud_probability, fraud_prediction,
                           route, feature_attribution   (challenge outputs)
    report.md              auto-generated write-up
    plots/*.png
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

T0 = time.time()

# keep BLAS / OpenMP from oversubscribing cores (nested with LightGBM / joblib);
# must run before numpy is imported anywhere downstream.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import yaml

from src.pipeline import Experiment

HERE = Path(__file__).parent


def apply_quick(cfg: dict) -> dict:
    """Shrink the config to a fast smoke-test configuration."""
    cfg["run"]["quick"] = True
    cfg["data"]["n_entities"] = 3200
    cfg["data"]["days"] = 160
    cfg["qrc"]["reservoir_seeds"] = [7]
    cfg["eval"]["bootstrap_iters"] = 400
    cfg["ablations"]["shots_sweep"] = [500]
    cfg["ablations"]["qubit_sweep"] = [4]
    cfg["ablations"]["K_sweep"] = [4]
    cfg["ablations"]["dt_sweep"] = [0.4, 1.5]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--data", choices=["synthetic", "ieee"], default="synthetic",
                    help="synthetic generator (default) or the real IEEE-CIS csv files")
    ap.add_argument("--outdir", default=None, help="override run.outdir")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.quick:
        cfg = apply_quick(cfg)
    cfg["run"]["data_source"] = args.data
    if args.data == "ieee" and args.quick and not cfg["data"].get("ieee_subsample"):
        cfg["data"]["ieee_subsample"] = 40000

    outdir = HERE / (args.outdir or cfg["run"]["outdir"])
    Experiment(cfg, outdir, t0=T0).run()


if __name__ == "__main__":
    main()
