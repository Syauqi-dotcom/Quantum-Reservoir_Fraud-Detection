#!/bin/bash
# SLURM job: QRC/reservoir hyperparameter search (tune_qrc.py) on the UID/D1n
# entity-key variant, on the Quasi cluster. Submit from THIS directory (code/):
#   sbatch run_quasi_tune.slurm
#
# Reuses the venv AND the IEEE-CIS dataset already deployed for job 119
# (experiments/ieee_uid) instead of rebuilding/recopying them -- see
# ../DEPLOY_QUASI_TUNE.md. Override either path with an env var if your
# layout differs:
#   sbatch --export=SHARED_VENV=/other/path,SHARED_DATASET=/other/path run_quasi_tune.slurm

#SBATCH --job-name="QRC tune"
#SBATCH --partition=qdisk        # confirm with `sinfo` if this differs on your account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00          # ~3-4h expected for TRIALS=40 + TOP_K=5 confirm at full scale (n_qubits capped at 6, see tune_qrc.py); padded for a shared/slower node
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job started on node: $(hostname)"
echo "Working dir: $(pwd)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-unset}"

SHARED_VENV="${SHARED_VENV:-/clusterfs/syauqi/abstraction_ieee_uid/code/.venv}"
SHARED_DATASET="${SHARED_DATASET:-/clusterfs/syauqi/abstraction_ieee_uid/code/dataset}"

if [ -x "$SHARED_VENV/bin/python" ]; then
  echo "Reusing existing venv: $SHARED_VENV"
  PY="$SHARED_VENV/bin/python"
else
  echo "No shared venv at $SHARED_VENV -- building a local one instead."
  if [ ! -d .venv ]; then
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -r requirements.txt
  fi
  PY="./.venv/bin/python"
fi

if [ ! -d dataset ]; then
  if [ -d "$SHARED_DATASET" ]; then
    echo "Symlinking dataset from $SHARED_DATASET (no re-copy)"
    ln -s "$SHARED_DATASET" dataset
  else
    echo "ERROR: no dataset/ here and no shared dataset at $SHARED_DATASET." >&2
    echo "Place the extracted IEEE-CIS csvs at code/dataset/<...IEEE-CIS...>/train_transaction.csv" >&2
    exit 1
  fi
fi

TRIALS="${TRIALS:-40}"
TOP_K="${TOP_K:-5}"
METHOD="${METHOD:-random}"

echo "Running QRC hyperparameter search (method=$METHOD trials=$TRIALS top_k=$TOP_K) ..."
"$PY" tune_qrc.py --data ieee --method "$METHOD" --trials "$TRIALS" --top-k "$TOP_K" --outdir ..

echo "Done. Results in $(realpath ..)  (tune_results.csv, tune_top.json, tuned_config.yaml)"
