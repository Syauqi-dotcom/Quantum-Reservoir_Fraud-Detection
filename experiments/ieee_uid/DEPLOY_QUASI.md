# Deploy to Quasi cluster

Self-contained experiment: **IEEE-CIS full run, UID/D1n entity key** (`card1 +
addr1 + D1n` instead of the `experiments/ieee_full/` baseline's `card1 +
card2 + card3 + addr1`). Change is isolated to `code/src/data_ieee.py`
(computes `D1n`) and `code/config.yaml` (`data.entity_keys.E2`) — nothing
else in the pipeline was touched. See `code/src/data_ieee.py`'s comment next
to the `D1n` computation for the rationale.

Everything needed to run standalone lives under `code/`: `src/`,
`run_experiment.py`, `config.yaml`, `requirements.txt`, `run_quasi.slurm`.

> **Field notes from the first real deployment (job 118, 2026-09-13):**
> - `rsync` is **not installed** on this cluster and there's no sudo to add
>   it — use `tar` piped over `ssh` instead (examples below).
> - `/job/<user>/...` is **per-node local disk**, NOT shared — a job that
>   `cd`s there on whatever node SLURM allocates will find it empty. Use
>   `/clusterfs/<user>/...` (confirmed NFS-mounted identically on every
>   node via `mount`) for anything a submitted job needs to read.
> - `sacct`/`sstat` don't work here (`AccountingStorageType`/
>   `JobAcctGatherType` are both unset in `slurm.conf` — a cluster-admin-level
>   config, not fixable per-user). For live CPU/memory, SSH to the node
>   SLURM allocated (`squeue` shows it) and run `ps`/`free` there directly
>   (see step 5).
> - The only partition seen so far is `qdisk` (matches `run_quasi.slurm`'s
>   default) — still worth an `sinfo` check on a different account/cluster
>   state.

## 1. Confirm the cluster's real partition name

```bash
ssh <you>@quasi09 sinfo
```

Edit `code/run_quasi.slurm`'s `#SBATCH --partition=` line if it differs from
`qdisk`.

## 2. Copy this folder to the cluster's shared filesystem

No `rsync` on this cluster — tar-over-ssh instead (also sidesteps quoting
issues with the dataset folder name's parentheses/spaces):

```bash
ssh <you>@quasi09 'mkdir -p /clusterfs/<you>/abstraction_ieee_uid'
cd experiments/ieee_uid
tar czf - . | ssh <you>@quasi09 'tar xzf - -C /clusterfs/<you>/abstraction_ieee_uid'
```

## 3. Put the IEEE-CIS dataset next to the code

The loader auto-detects `code/dataset/*IEEE-CIS*/train_transaction.csv`.
The dataset is **not** in this folder (gitignored, ~1.3 GB) — copy it
separately, e.g. from wherever it's already extracted on this machine:

```bash
ssh <you>@quasi09 'mkdir -p "/clusterfs/<you>/abstraction_ieee_uid/code/dataset/(Primary) IEEE-CIS Fraud Detection"'
cd "dataset/(Primary) IEEE-CIS Fraud Detection"
tar czf - . | ssh <you>@quasi09 'tar xzf - -C "/clusterfs/<you>/abstraction_ieee_uid/code/dataset/(Primary) IEEE-CIS Fraud Detection"'
```

(or re-download+unzip the Kaggle files directly on the cluster into that
path — either works, the loader just globs `dataset/*IEEE-CIS*`.)

## 4. Build the venv and submit the job

Build it explicitly first rather than relying on `run_quasi.slurm`'s
auto-create — compute nodes may have no internet access, so build it on the
login node (`quasi09`) where `pip install` can actually reach PyPI:

```bash
ssh <you>@quasi09
cd /clusterfs/<you>/abstraction_ieee_uid/code
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -c "import lightgbm, catboost, sklearn, pennylane, shap; print('imports OK')"
sbatch run_quasi.slurm
```

Note the job ID `sbatch` prints (e.g. `Submitted batch job 118`).

## 5. Monitor

```bash
squeue -u <you>                                  # job state (R/PD/...) + which node it landed on
tail -f /clusterfs/<you>/abstraction_ieee_uid/code/slurm-<jobid>.out   # pipeline progress
tail -f /clusterfs/<you>/abstraction_ieee_uid/code/slurm-<jobid>.err   # should stay empty
```

For live CPU/memory (`sacct`/`sstat` don't work here — see field notes
above), SSH to whichever node `squeue` shows and read `ps` directly:

```bash
ssh <you>@quasi09 ssh <node-from-squeue> \
  "ps -u <you> -o pid,%cpu,%mem,rss,vsz,etime,cmd --sort=-rss | head; free -h"
```

Historical full IEEE-CIS run (same pipeline, old entity key) took ~3.7h on a
laptop CPU; `run_quasi.slurm` requests an 8h wall-clock budget for headroom
on a shared/slower node. Adjust `--time` / `--cpus-per-task` if you know the
node spec.

## 6. Bring results back

Output lands one level above `code/` (same layout as `experiments/ieee_full/`
and `experiments/ieee_smoke/`):

```bash
ssh <you>@quasi09 'cd /clusterfs/<you>/abstraction_ieee_uid && tar czf - manifest.json metrics.json success_criteria.json report.md predictions.csv plots run_full.log' \
  | tar xzf - -C experiments/ieee_uid/
```

## 7. What to check first

Compare against `experiments/ieee_full/metrics.json` (baseline entity key):

- `ablations.shuffle_sequence` and `ablations.no_history_K1` — in the
  baseline both were *helpful* (shuffling/removing history improved band
  AUPRC), the opposite of a real temporal signal. A local 40k/1-seed smoke
  test of this exact code flipped both to *harmful* (see conversation log) —
  check whether that holds on the full 590k/2-seed run.
- `deltas.c1_auprc_Q3_vs_B4_full` and `deltas.c3_auprc_Q2_vs_ESN_band` — do
  they move toward/past 0 vs the baseline's −0.0151 / −0.0039?
- `success_criteria.json` — does the overall verdict change from `NO`?
