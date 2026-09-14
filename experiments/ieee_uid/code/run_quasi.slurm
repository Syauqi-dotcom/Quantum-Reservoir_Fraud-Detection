#!/bin/bash
# SLURM job for the Abstraction PoC (IEEE-CIS full run, UID/D1n entity key
# variant) on the Quasi cluster. Submit from THIS directory (code/):
#   sbatch run_quasi.slurm
#
# Adapted from the Quasi job-submission template:
# https://github.com/BRIN-Q/tacit-knowledge/blob/main/Tutorials/QuasiCluster-construction.md
# That template is written for an MPI/Quantum-ESPRESSO workload; this job is a
# single-process Python pipeline, so there is no mpirun / multi-node step.

#SBATCH --job-name=abstraction_ieee_uid
#SBATCH --partition=qdisk        # TODO: confirm the real partition name with `sinfo` on Quasi -- qdisk is the only one named in the tutorial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00          # historical full IEEE-CIS run took ~3.7h on the dev laptop; padded for a shared/slower node
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job started on node: $(hostname)"
echo "Working dir: $(pwd)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-unset}"

if [ ! -d dataset ]; then
  echo "ERROR: no dataset/ found next to this script." >&2
  echo "Place the extracted IEEE-CIS csvs at code/dataset/<...IEEE-CIS...>/train_transaction.csv" >&2
  echo "(see ../DEPLOY_QUASI.md)." >&2
  exit 1
fi

# one-time environment setup -- safe to re-run, skips if venv already exists.
# NOTE: if Quasi's compute nodes have no internet access, build this venv on
# the login node first (same partition filesystem), then just submit the job.
if [ ! -d .venv ]; then
  echo "Creating venv..."
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

echo "Running Abstraction PoC (IEEE-CIS full, UID/D1n entity key) ..."
./.venv/bin/python run_experiment.py --data ieee --outdir ..

echo "Done. Results in $(realpath ..)"
