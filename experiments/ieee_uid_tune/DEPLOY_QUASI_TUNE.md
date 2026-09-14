# Deploy to Quasi cluster - QRC hyperparameter search

Self-contained: same UID/D1n entity-key code as `experiments/ieee_uid/`, plus
`code/tune_qrc.py` (random search over Stage D reservoir hyperparameters +
Stage C history window `K`, scored on the validation band, never the frozen
test split -- see the docstring in `tune_qrc.py` for why).

Why this run exists: job 119 (`experiments/ieee_uid/`) came back verdict
`NO` for every criterion except C3 (parity with the classical ESN). Its own
ablations showed the *frozen defaults* were the problem, not the entity key:
a random reservoir beat the designed one by +0.02 AUPRC, 4 qubits beat 6,
K=4 beat K=8, dt=0.4 beat 0.75. This job searches that space properly instead
of hand-picking one ablation at a time.

Everything needed lives under `code/`: `src/`, `run_experiment.py`,
`tune_qrc.py`, `config.yaml`, `requirements.txt`, `run_quasi_tune.slurm`.

## Reuses job 119's venv and dataset -- no rebuild needed

`run_quasi_tune.slurm` points at the venv and dataset already deployed for
job 119 by absolute path:

```
SHARED_VENV=/clusterfs/syauqi/abstraction_ieee_uid/code/.venv
SHARED_DATASET=/clusterfs/syauqi/abstraction_ieee_uid/code/dataset
```

If `$SHARED_VENV/bin/python` exists it's used directly (no `pip install`
step); the dataset is symlinked in rather than re-copied (~1.3 GB saved). If
job 119's folder isn't there anymore, override both paths at submit time:

```bash
sbatch --export=SHARED_VENV=/other/path,SHARED_DATASET=/other/path run_quasi_tune.slurm
```

or fall back to the full `experiments/ieee_uid/DEPLOY_QUASI.md` steps
(build a fresh venv, copy the dataset) if neither shared copy exists.

## 1. Copy this folder to the cluster's shared filesystem

No `rsync` on Quasi (see field notes in `../ieee_uid/DEPLOY_QUASI.md`) --
tar-over-ssh instead:

```bash
ssh <you>@quasi09 'mkdir -p /clusterfs/<you>/abstraction_qrc_tune'
cd experiments/ieee_uid_tune
tar czf - . | ssh <you>@quasi09 'tar xzf - -C /clusterfs/<you>/abstraction_qrc_tune'
```

## 2. Confirm the partition, then submit

```bash
ssh <you>@quasi09 sinfo    # edit run_quasi_tune.slurm's --partition= if it isn't qdisk
ssh <you>@quasi09
cd /clusterfs/<you>/abstraction_qrc_tune/code
sbatch run_quasi_tune.slurm
```

Optional overrides via `--export` (SLURM env vars the script reads):
`TRIALS` (default 40), `TOP_K` (default 5), `METHOD` (`random` or `grid`,
default `random`), e.g.:

```bash
sbatch --export=TRIALS=60,TOP_K=8 run_quasi_tune.slurm
```

Runtime budget: `n_qubits` is capped at 6 in `tune_qrc.py`'s default search
space (the exact density-matrix simulator costs ~dim^3 = 8^n_qubits per
evolution step, so 7-8 qubits are ~8-64x slower per trial than 6 -- explore
those separately with `--subsample` if wanted). At the full IEEE-CIS band
scale, each trial costs roughly 1-4 minutes; TRIALS=40 + TOP_K=5 confirmation
should land around 2.5-4h, well inside the 8h `--time` budget.

## 3. Monitor

```bash
squeue -u <you>
tail -f /clusterfs/<you>/abstraction_qrc_tune/code/slurm-<jobid>.out
```

## 4. Bring results back

```bash
ssh <you>@quasi09 'cd /clusterfs/<you>/abstraction_qrc_tune && tar czf - tune_results.csv tune_top.json tuned_config.yaml' \
  | tar xzf - -C experiments/ieee_uid_tune/
```

## 5. What to do with the result

`tuned_config.yaml` is a full copy of `config.yaml` with the winning
`qrc.*` values and `sequence.K` swapped in. `tune_top.json` shows whether the
best candidate's validation-band AUPRC actually clears classical `p0` with a
95% CI (`delta_ci_low > 0`) -- if not, the search didn't find a config that
beats "no QRC at all" on this entity key, which is itself a useful (honest)
result.

If it does clear that bar, run ONE honest confirmation on the frozen test
split (this is the only time this config touches test):

```bash
cd code
"$SHARED_VENV/bin/python" ../../run_experiment.py \
  --config ../tuned_config.yaml --data ieee --outdir ../../ieee_uid_tuned
```

(adjust paths to wherever you copy `tuned_config.yaml` and
`run_experiment.py` together -- easiest is to drop `tuned_config.yaml` next
to `run_experiment.py` in a copy of `code/` and run from there, same as
`experiments/ieee_uid/DEPLOY_QUASI.md` step 4).
