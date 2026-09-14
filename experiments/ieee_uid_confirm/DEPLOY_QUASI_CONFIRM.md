# Deploy to Quasi cluster - ONE frozen-test-split confirmation

This is the final step of `../ieee_uid_tune/DEPLOY_QUASI_TUNE.md` (step 5):
run the winning hyperparameters from `tune_top.json` (already baked into
`code/config.yaml`, a copy of `tuned_config.yaml`) through the pipeline
`--data ieee` with no `--quick`, so it touches the FROZEN test split. Do
this only once with this config -- if it fails, that's the honest answer,
not a cue to go back and search more.

Winning config confirmed here: n_qubits=4, input_qubits=3,
reupload_rounds=3, virtual_nodes=2, dt=0.25, J=0.5, h=0.1, K=6 -- validation-
band AUPRC 0.204 mean, beat classical `p0` with `delta_ci_low > 0` on both
seeds (see `../ieee_uid_tune/tune_top.json`).

Everything needed lives under `code/`: `src/`, `run_experiment.py`,
`config.yaml` (= tuned_config.yaml), `requirements.txt`,
`run_quasi_confirm.slurm`.

## Reuses job 119's venv and dataset -- no rebuild needed

`run_quasi_confirm.slurm` points at the venv and dataset already deployed
for job 119 by absolute path:

```
SHARED_VENV=/clusterfs/syauqi/abstraction_ieee_uid/code/.venv
SHARED_DATASET=/clusterfs/syauqi/abstraction_ieee_uid/code/dataset
```

Override both at submit time if job 119's folder is gone:

```bash
sbatch --export=SHARED_VENV=/other/path,SHARED_DATASET=/other/path run_quasi_confirm.slurm
```

## 1. Copy this folder to the cluster's shared filesystem

```bash
ssh <you>@quasi09 'mkdir -p /clusterfs/<you>/abstraction_qrc_confirm'
cd experiments/ieee_uid_confirm
tar czf - . | ssh <you>@quasi09 'tar xzf - -C /clusterfs/<you>/abstraction_qrc_confirm'
```

## 2. Confirm the partition, then submit

```bash
ssh <you>@quasi09 sinfo    # edit run_quasi_confirm.slurm's --partition= if it isn't qdisk
ssh <you>@quasi09
cd /clusterfs/<you>/abstraction_qrc_confirm/code
sbatch run_quasi_confirm.slurm
```

## 3. Monitor

```bash
squeue -u <you>
tail -f /clusterfs/<you>/abstraction_qrc_confirm/code/slurm-<jobid>.out
```

## 4. Bring results back

```bash
ssh <you>@quasi09 'cd /clusterfs/<you>/abstraction_qrc_confirm && tar czf - metrics.json success_criteria.json manifest.json report.md predictions.csv plots' \
  | tar xzf - -C experiments/ieee_uid_confirm/
```

## 5. Read the verdict

`success_criteria.json` and `report.md` say pass/fail per pre-registered
criterion on the real test split -- this is the number that answers
"does the tuned hybrid QRC actually beat the classical baseline," not the
validation-band number from the tuning search.
