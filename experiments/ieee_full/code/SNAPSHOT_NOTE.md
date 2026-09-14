# Code snapshot

`src/`, `run_experiment.py`, `config.yaml` here are a copy of the pipeline code
at commit `3e7bc37` (this repo's only code version — both the `ieee_full` and
`ieee_smoke` runs predate the first commit, so there is no earlier revision to
diff against).

The **same code** produced every experiment in `experiments/`; nothing was
forked or changed per experiment. What differs between runs is config values
and the `--data` flag, and the exact effective config actually used for *this*
run (after `--quick`/`--data` overrides, plus the FROZEN thresholds learned on
validation) is recorded in full in `../manifest.json` under the `"config"`
key — treat that as the source of truth over the `config.yaml` copied here,
which is the current repo template and may have drifted (e.g. `run.outdir`)
since this run was produced.

To rerun standalone on another machine: copy this whole experiment folder,
`cd` into `code/`, `pip install -r requirements.txt` (from the main repo) and
run `python run_experiment.py --data ieee --outdir ..`.
