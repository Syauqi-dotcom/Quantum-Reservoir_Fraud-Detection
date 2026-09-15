# Deploy to Quasi cluster - Amazon Braket cross-check

Tiny, self-contained: just `src/qrc.py` (the reservoir this checks against)
and `src/qrc_braket.py` (builds the same single reservoir step as a Braket
`Circuit`, runs it on the Braket SDK's own `LocalSimulator` -- no AWS
account/credentials needed) plus a minimal `requirements-braket.txt`.

Deliberately its own fresh venv, NOT the shared one at
`/clusterfs/syauqi/abstraction_ieee_uid/code/.venv` used by job 119 / the
tune / confirm jobs -- this avoids touching the numpy/scipy pins those jobs
depend on. This check needs nothing else from `requirements-braket.txt`'s
package set and runs in well under a minute, so no SLURM job is really
needed; run it directly on the login node.

## 1. Copy this folder to the cluster's shared filesystem

```bash
ssh <you>@quasi09 'mkdir -p /clusterfs/<you>/abstraction_qrc_braket'
cd experiments/qrc_braket_check
tar czf - . | ssh <you>@quasi09 'tar xzf - -C /clusterfs/<you>/abstraction_qrc_braket'
```

## 2. Build a small venv and run

```bash
ssh <you>@quasi09
cd /clusterfs/<you>/abstraction_qrc_braket/code
python3 -m venv .venv
./.venv/bin/pip install -r requirements-braket.txt   # needs internet on quasi09 (login node)
./.venv/bin/python -m src.qrc_braket
```

Expected output (numbers only -- values are illustrative, they depend on
the fixed demo `feats`/`seed` in `qrc_braket.py`'s `verify()`):

```
numpy         <Z_i> = [ 0.36501 -0.08493  0.66693  0.80404]
amazon braket <Z_i> = [ 0.36501 -0.08493  0.66693  0.80404]
max abs diff        = 1.11e-15 -> OK
```

`OK` means Braket's own SDK/simulator reproduces the exact-diagonalisation
NumPy result to machine precision -- confirms the reservoir circuit is
correctly expressed in Braket's gate model, not just "portable in
principle."

If `quasi09` (login node) has no outbound internet either, `pip install`
will fail -- in that case build this same venv on a machine that does have
internet (e.g. locally) and `rsync`/tar the resulting `.venv/` across
instead of trying to install offline.

## 3. (Optional) point at a real AWS Braket simulator/QPU instead

`qrc_braket.py`'s `_braket_single_step` uses `LocalSimulator()`. Swapping
that for e.g.
`AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")` runs
the identical circuit on AWS's on-demand simulator instead -- needs AWS
credentials configured on whichever machine runs it (`aws configure`, or
env vars), and costs a small amount per task. Not required for this
cross-check to "count" as using Amazon Braket -- `LocalSimulator` is part
of the same `amazon-braket-sdk` and is the documented way to develop/test
Braket circuits before spending on a real device.
