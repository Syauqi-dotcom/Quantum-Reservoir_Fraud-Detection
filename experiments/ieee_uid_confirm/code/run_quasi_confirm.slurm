#!/bin/bash
# SLURM job: ONE honest confirmation run of the tuned QRC hyperparameters
# (from experiments/ieee_uid_tune/tuned_config.yaml, baked in here as
# config.yaml) on the frozen IEEE-CIS test split. Submit from THIS
# directory (code/):
#   sbatch run_quasi_confirm.slurm
#
# This is the only time tuned_config.yaml is allowed to touch the test
# split -- see ../../ieee_uid_tune/DEPLOY_QUASI_TUNE.md step 5.
#
# Reuses the venv AND dataset already deployed for job 119
# (experiments/ieee_uid) -- override either path if your layout differs:
#   sbatch --export=SHARED_VENV=/other/path,SHARED_DATASET=/other/path run_quasi_confirm.slurm

#SBATCH --job-name="QRC run"
#SBATCH --partition=qdisk        # confirm with `sinfo` if this differs on your account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00          # single full-scale run; historical full IEEE-CIS run ~3.7h on dev laptop, padded
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

echo "Running ONE confirmation of tuned QRC config on the frozen test split ..."
"$PY" run_experiment.py --config config.yaml --data ieee --outdir ..

echo "Done. Results in $(realpath ..) (metrics.json, success_criteria.json, report.md, predictions.csv, plots/)"
