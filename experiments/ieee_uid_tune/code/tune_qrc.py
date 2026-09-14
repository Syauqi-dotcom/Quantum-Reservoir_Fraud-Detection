#!/usr/bin/env python3
"""
Hyperparameter search for Stage D (quantum reservoir) + Stage E (fusion),
targeting the validation-band AUPRC.

Why validation, never test
---------------------------
`experiments/*/report.md` treats the chronological test split as FROZEN and
only ever used once, at the end, to compute the pre-registered success
criteria (see `evaluate_success_criteria` in src/pipeline.py). Model
selection (which fusion head to deploy, alpha* for smoothing, tau_low/high
for routing) is always done on validation. This script follows the same
rule: every candidate is scored on the validation band, so searching over
many QRC configs here never leaks into the number that finally gets
reported. Run `run_experiment.py` once at the end with the winning config to
get the honest, bootstrap-CI'd, multi-seed test-set numbers.

What it does
------------
1. Runs Stages A-C (data, GBDT baselines B0/B1/B4/B5/CatBoost, calibration,
   router, per-entity band sequences) exactly once -- this is the expensive,
   hyperparameter-independent part of the pipeline.
2. For each candidate QRC/reservoir config, builds the embedding
   (`src/qrc.py::make_embeddings`), PCA-reduces it, fits the `feature_fusion`
   read-out (guard disabled, same convention as the ablations in
   `src/pipeline.py::run_ablations`), and scores band AUPRC on validation.
3. Ranks candidates, re-scores the top-K with the full multi-seed list and a
   paired-bootstrap delta vs the classical p0 (does the tuned reservoir beat
   "no QRC at all" with a CI, on validation), and writes:
     <outdir>/tune_results.csv         every trial, sorted best-first
     <outdir>/tune_top.json            top-K with bootstrap CIs
     <outdir>/tuned_config.yaml        full config.yaml with qrc/sequence.K
                                        replaced by the winning candidate
     <outdir>/plots/search_progress.png     AUPRC per trial vs current-config / p0 reference
     <outdir>/plots/param_sensitivity.png   AUPRC vs each hyperparameter axis (8 panels)
     <outdir>/plots/top_trials.png          bar chart of the best trials vs reference
     <outdir>/plots/confirm_per_seed.png    top-K candidates, grouped bars per seed
   (pass --no-plots to skip; a plotting failure never drops already-written
   results, same convention as src/pipeline.py::_plots)

Motivation for the search space (from experiments/ieee_uid ablations, job
119): a random reservoir beat the designed Ising one by +0.02 AUPRC,
qubits=4 beat the default 6, K=4 beat the default 8, and dt=0.4 beat the
default 0.75 -- i.e. the frozen defaults were past the sweet spot on that
entity key. This script automates finding a better point instead of
hand-picking one more ablation at a time.

Usage
-----
    # quick local search on a subsample (few minutes)
    python tune_qrc.py --data ieee --subsample 60000 --trials 40

    # full search (run this on the Quasi cluster, same as run_quasi.slurm)
    python tune_qrc.py --data ieee --trials 80 --top-k 5 \
        --outdir experiments/ieee_uid_tune

    # then confirm honestly on the frozen test split:
    python run_experiment.py --config experiments/ieee_uid_tune/tuned_config.yaml \
        --data ieee --outdir experiments/ieee_uid_tuned
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

T0 = time.time()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import yaml

from src import evaluate as ev
from src.pipeline import EmbReducer, Experiment, fit_fusion_variant
from src.qrc import make_embeddings

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# search space: every axis the job-119 ablations flagged as away from optimal,
# plus the neighbours of the frozen defaults so the search can also confirm
# the defaults were fine (override any of this with --space a JSON file).
#
# n_qubits is capped at 6 by default: the simulator is an exact density-matrix
# sim (src/qrc.py), so cost per evolution step scales ~ dim^3 = 8^n_qubits.
# Going from 6 to 8 qubits is a ~64x slowdown per trial -- on the FULL
# IEEE-CIS band (~job 119 scale) one 6-qubit trial already costs ~3-4 min, so
# 40 trials at 8 qubits would blow well past an 8h SLURM budget. Search 7/8
# qubits separately with --subsample if you want to explore that axis.
# ---------------------------------------------------------------------------
DEFAULT_SPACE = dict(
    n_qubits=[3, 4, 5, 6],
    input_qubits=[2, 3, 4],
    reupload_rounds=[2, 3, 4],
    virtual_nodes=[2, 3, 4],
    dt=[0.25, 0.4, 0.5, 0.75, 1.0, 1.5],
    J=[0.5, 1.0, 1.5, 2.0, 3.0],
    h=[0.1, 0.3, 0.5, 0.8, 1.2],
    K=[3, 4, 5, 6, 8],
)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ===========================================================================
# stage A-C once, shared by every trial
# ===========================================================================
def build_base_experiment(cfg: dict, outdir: Path) -> Experiment:
    exp = Experiment(cfg, outdir)
    log("stage A-C (data, GBDT baselines, router, band sequences) ...")
    exp.load_data()
    exp.split_and_features()
    exp.train_baselines()
    exp.entity_smoothing()
    exp.fit_router()
    exp.build_band_sequences()
    exp.tr, exp.va, exp.te = exp.split_idx
    exp.kdim = int(cfg["fusion"].get("embed_pca_dim", 28))
    exp.Cg = cfg["fusion"]["C_grid"]
    exp.seeds = cfg["qrc"]["reservoir_seeds"]
    log(f"  band sizes: train={len(exp.tr)} val={len(exp.va)} test={len(exp.te)}  "
        f"embed target dim={exp.kdim}")
    return exp


def trim_to_K(mask_all: np.ndarray, K: int | None) -> np.ndarray:
    """Zero the mask beyond the most recent K steps per row -- mirrors the
    K_sweep ablation in src/pipeline.py::run_ablations."""
    if K is None:
        return mask_all
    m = mask_all.copy()
    for i in range(len(m)):
        idx = np.where(m[i] > 0)[0]
        if len(idx) > K:
            m[i, idx[:-K]] = 0.0
    return m


def score_candidate(exp: Experiment, qcfg: dict, K: int, seed: int):
    """Build the embedding for one candidate with ONE reservoir seed, fit
    feature_fusion (unguarded, same convention as the ablations), return
    val-band AUPRC + the val predictions.

    NOTE: `make_embeddings` picks up to `qcfg["spatial_multiplex"]` seeds
    from the list it's given -- passing more seeds than that is silently a
    no-op (they're dropped, not averaged). So multi-seed robustness is
    handled by calling this once per seed and combining results explicitly
    (see `confirm_top`), never by handing it a longer seed list."""
    mask = trim_to_K(exp.mask_all, K)
    qcfg1 = {**qcfg, "spatial_multiplex": 1}
    Z = make_embeddings(qcfg1, [seed], exp.seq_all, mask)
    Zr = EmbReducer(exp.kdim).fit_transform(Z, exp.tr)
    _, pv, _ = fit_fusion_variant(
        "feature_fusion", exp.Cg, exp.p0_all_band, Zr, exp.ctx_all,
        exp.y_all, exp.split_idx, guard=False)
    auprc = float(ev.auprc_fn(exp.y_all[exp.va], pv))
    return auprc, pv


# ===========================================================================
# candidate sampling
# ===========================================================================
def sample_candidate(rng: np.random.Generator, space: dict, n_features: int) -> dict | None:
    n_qubits = int(rng.choice(space["n_qubits"]))
    input_qubits = int(min(rng.choice(space["input_qubits"]), n_qubits - 1))
    if input_qubits < 1:
        return None
    reupload_rounds = int(rng.choice(space["reupload_rounds"]))
    if input_qubits * reupload_rounds < n_features:
        return None  # would silently truncate step features -- not a fair trial
    return dict(
        n_qubits=n_qubits,
        input_qubits=input_qubits,
        reupload_rounds=reupload_rounds,
        virtual_nodes=int(rng.choice(space["virtual_nodes"])),
        dt=float(rng.choice(space["dt"])),
        J=float(rng.choice(space["J"])),
        h=float(rng.choice(space["h"])),
        K=int(rng.choice(space["K"])),
    )


def random_search(exp: Experiment, base_qrc: dict, space: dict, n_features: int,
                   trials: int, seed: int, search_seed: int):
    rng = np.random.default_rng(seed)
    seen = set()
    results = []
    tried = 0
    skipped = 0
    while tried < trials:
        cand = sample_candidate(rng, space, n_features)
        if cand is None:
            skipped += 1
            if skipped > trials * 20:
                log("  too many infeasible samples (input_qubits*reupload_rounds < "
                    "n_step_features) -- widen --space")
                break
            continue
        key = tuple(sorted(cand.items()))
        if key in seen:
            continue
        seen.add(key)
        tried += 1
        qcfg = {**base_qrc, **{k: v for k, v in cand.items() if k != "K"}}
        t0 = time.time()
        try:
            auprc, _ = score_candidate(exp, qcfg, cand["K"], seed=search_seed)
        except Exception as e:
            log(f"  [{tried:3d}/{trials}] FAILED {cand} -> {e!r}")
            continue
        secs = time.time() - t0
        row = {**cand, "val_band_auprc": auprc, "seconds": round(secs, 2)}
        results.append(row)
        log(f"  [{tried:3d}/{trials}] val-band AUPRC={auprc:.4f}  "
            f"qubits={cand['n_qubits']} in={cand['input_qubits']} "
            f"R={cand['reupload_rounds']} V={cand['virtual_nodes']} "
            f"dt={cand['dt']} J={cand['J']} h={cand['h']} K={cand['K']}  "
            f"({secs:.1f}s)")
    return results


def grid_search(exp: Experiment, base_qrc: dict, space: dict, n_features: int,
                 search_seed: int):
    import itertools
    keys = ["n_qubits", "input_qubits", "reupload_rounds", "virtual_nodes",
            "dt", "J", "h", "K"]
    combos = list(itertools.product(*(space[k] for k in keys)))
    log(f"grid search: {len(combos)} raw combinations before feasibility filter")
    results = []
    tried = 0
    for combo in combos:
        cand = dict(zip(keys, combo))
        cand["input_qubits"] = min(cand["input_qubits"], cand["n_qubits"] - 1)
        if cand["input_qubits"] < 1:
            continue
        if cand["input_qubits"] * cand["reupload_rounds"] < n_features:
            continue
        tried += 1
        qcfg = {**base_qrc, **{k: v for k, v in cand.items() if k != "K"}}
        t0 = time.time()
        try:
            auprc, _ = score_candidate(exp, qcfg, cand["K"], seed=search_seed)
        except Exception as e:
            log(f"  [{tried}] FAILED {cand} -> {e!r}")
            continue
        secs = time.time() - t0
        row = {**cand, "val_band_auprc": auprc, "seconds": round(secs, 2)}
        results.append(row)
        log(f"  [{tried}] val-band AUPRC={auprc:.4f}  {cand}  ({secs:.1f}s)")
    return results


# ===========================================================================
# confirm top-K with full seed list + bootstrap CI vs classical p0 (val only)
# ===========================================================================
def confirm_top(exp: Experiment, base_qrc: dict, top_rows: list, cfg: dict):
    """Re-score each top candidate once per reservoir seed (mirrors
    src/pipeline.py::multi_seed_check / the C4 success criterion: a result
    only counts as robust if it holds for EVERY seed, not just the one the
    search happened to use)."""
    log(f"confirming top {len(top_rows)} candidates across all "
        f"{len(exp.seeds)} reservoir seeds {exp.seeds} (bootstrap CI vs p0, "
        f"still validation only) ...")
    bi = cfg["eval"]["bootstrap_iters"]
    p0_val_band = exp.p0_all_band[exp.va]
    y_val_band = exp.y_all[exp.va]
    confirmed = []
    for i, row in enumerate(top_rows):
        cand = {k: row[k] for k in
                ("n_qubits", "input_qubits", "reupload_rounds", "virtual_nodes",
                 "dt", "J", "h", "K")}
        qcfg = {**base_qrc, **{k: v for k, v in cand.items() if k != "K"}}
        per_seed = []
        for s in exp.seeds:
            auprc, pv = score_candidate(exp, qcfg, cand["K"], seed=s)
            delta = ev.paired_bootstrap_delta(y_val_band, pv, p0_val_band, ev.auprc_fn,
                                              iters=bi, seed=cfg["run"]["seed"])
            per_seed.append(dict(seed=int(s), val_band_auprc=auprc,
                                 delta_vs_p0=delta["delta"],
                                 delta_ci_low=delta["ci_low"],
                                 delta_ci_high=delta["ci_high"]))
            log(f"  #{i + 1} seed={s}: val-band AUPRC={auprc:.4f}  "
                f"delta vs p0={delta['delta']:+.4f} "
                f"CI=[{delta['ci_low']:+.4f},{delta['ci_high']:+.4f}]")
        mean_auprc = float(np.mean([r["val_band_auprc"] for r in per_seed]))
        holds_all_seeds = all(r["delta_ci_low"] > 0 for r in per_seed)
        confirmed.append({**row, "per_seed": per_seed,
                          "val_band_auprc_mean": mean_auprc,
                          "holds_across_all_seeds": holds_all_seeds})
        log(f"  #{i + 1} summary: mean val-band AUPRC={mean_auprc:.4f}  "
            f"holds_across_all_seeds={holds_all_seeds}  {cand}")
    confirmed.sort(key=lambda r: (r["holds_across_all_seeds"], r["val_band_auprc_mean"]),
                   reverse=True)
    return confirmed


# ===========================================================================
# plots
# ===========================================================================
def write_tune_plots(outdir: Path, results: list, confirmed: list,
                     baseline_auprc: float, p0_auprc: float):
    """Same convention as src/pipeline.py::_plots -- a plotting failure never
    drops the already-written CSV/JSON/YAML results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as mplt

        from src import plots as plt_

        plots_dir = outdir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # 1. search progress: every trial, in the order it was tried
        fig, ax = mplt.subplots(figsize=(7, 4))
        idx = list(range(1, len(results) + 1))
        vals = [r["val_band_auprc"] for r in results]
        ax.scatter(idx, vals, s=18, alpha=0.7, color="#3b7dd8")
        ax.axhline(baseline_auprc, color="tab:orange", ls="--", lw=1,
                  label=f"current config.yaml ({baseline_auprc:.4f})")
        ax.axhline(p0_auprc, color="k", ls=":", lw=1,
                  label=f"classical p0, no QRC ({p0_auprc:.4f})")
        ax.set_xlabel("trial # (search order)")
        ax.set_ylabel("val-band AUPRC")
        ax.set_title("QRC hyperparameter search progress")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "search_progress.png", dpi=120)
        mplt.close(fig)

        # 2. per-hyperparameter sensitivity -- which axis actually matters
        params = ["n_qubits", "input_qubits", "reupload_rounds", "virtual_nodes",
                  "dt", "J", "h", "K"]
        fig, axes = mplt.subplots(2, 4, figsize=(16, 7))
        for ax, p in zip(axes.ravel(), params):
            xs = [r[p] for r in results]
            ys = [r["val_band_auprc"] for r in results]
            ax.scatter(xs, ys, s=14, alpha=0.6, color="#3b7dd8")
            ax.axhline(baseline_auprc, color="tab:orange", ls="--", lw=0.8)
            ax.set_xlabel(p)
            ax.set_ylabel("val-band AUPRC")
            ax.grid(alpha=0.3)
        fig.suptitle("Hyperparameter sensitivity (each dot = one trial; "
                     "dashed line = current config.yaml)")
        fig.tight_layout()
        fig.savefig(plots_dir / "param_sensitivity.png", dpi=120)
        mplt.close(fig)

        # 3. top trials vs reference (reuses src/plots.py's bar_compare, same
        # visual convention as run_experiment.py's band_auprc.png / ablations.png)
        top_n = min(10, len(results))
        names = [f"#{i + 1}" for i in range(top_n)]
        vals = [results[i]["val_band_auprc"] for i in range(top_n)]
        plt_.bar_compare(names, vals, "Top trials - validation-band AUPRC",
                         "AUPRC", str(plots_dir), "top_trials.png", ref=baseline_auprc)

        # 4. confirmed top-K, grouped bars per reservoir seed -- visually shows
        # robustness: similar bar heights across seeds = robust, very
        # different = the search got lucky on one seed (see C4 in the main
        # pipeline)
        if confirmed:
            fig, ax = mplt.subplots(figsize=(max(6, len(confirmed) * 1.6), 4))
            seeds = sorted({s["seed"] for r in confirmed for s in r["per_seed"]})
            width = 0.8 / max(1, len(seeds))
            xpos = np.arange(len(confirmed))
            for j, sd in enumerate(seeds):
                vals = [next(s["val_band_auprc"] for s in r["per_seed"] if s["seed"] == sd)
                       for r in confirmed]
                ax.bar(xpos + j * width - width * (len(seeds) - 1) / 2, vals, width,
                      label=f"seed {sd}")
            ax.axhline(baseline_auprc, color="k", ls="--", lw=1, label="current config.yaml")
            ax.set_xticks(xpos)
            ax.set_xticklabels([f"#{i + 1}" for i in range(len(confirmed))])
            ax.set_ylabel("val-band AUPRC")
            ax.set_title("Top candidates - per-seed confirmation")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(plots_dir / "confirm_per_seed.png", dpi=120)
            mplt.close(fig)

        log(f"  wrote plots -> {plots_dir}")
    except Exception as e:                        # pragma: no cover
        log(f"  plotting failed ({e!r}) -- results already written, skipping plots")


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--data", choices=["synthetic", "ieee"], default="ieee")
    ap.add_argument("--subsample", type=int, default=None,
                    help="systematic time-preserving subsample of IEEE-CIS for a fast "
                         "search (recommended locally; omit on the cluster for the real run)")
    ap.add_argument("--outdir", default="experiments/qrc_tune",
                    help="where trial logs / results / tuned_config.yaml are written")
    ap.add_argument("--method", choices=["random", "grid"], default="random")
    ap.add_argument("--trials", type=int, default=40, help="random search only")
    ap.add_argument("--sampler-seed", type=int, default=0,
                    help="RNG seed for candidate sampling (not a reservoir seed)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-plots", action="store_true",
                    help="skip writing plots/*.png (they're on by default)")
    ap.add_argument("--space", default=None,
                    help="path to a JSON file overriding DEFAULT_SPACE (same keys: "
                         "n_qubits, input_qubits, reupload_rounds, virtual_nodes, dt, J, h, K)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["run"]["data_source"] = args.data
    if args.subsample:
        cfg["data"]["ieee_subsample"] = args.subsample

    space = DEFAULT_SPACE
    if args.space:
        space = {**DEFAULT_SPACE, **json.load(open(args.space))}

    outdir = HERE / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    exp = build_base_experiment(cfg, outdir)
    base_qrc = {**cfg["qrc"]}
    n_features = len(cfg["sequence"]["step_features"])

    search_seed = exp.seeds[0]  # search itself always uses the primary reservoir
                                # seed for speed; confirm_top() checks every seed
    baseline_auprc, _ = score_candidate(
        exp, base_qrc, cfg["sequence"]["K"], seed=search_seed)
    p0_auprc = float(ev.auprc_fn(exp.y_all[exp.va], exp.p0_all_band[exp.va]))
    log(f"reference: classical p0 val-band AUPRC        = {p0_auprc:.4f}")
    log(f"reference: current config.yaml qrc val-band AUPRC = {baseline_auprc:.4f}  "
        f"(seed={search_seed})")

    t_search = time.time()
    if args.method == "random":
        results = random_search(exp, base_qrc, space, n_features,
                                args.trials, args.sampler_seed, search_seed)
    else:
        results = grid_search(exp, base_qrc, space, n_features, search_seed)
    log(f"search done in {time.time() - t_search:.1f}s  ({len(results)} valid trials)")

    if not results:
        raise SystemExit("no valid candidates were scored -- widen --space "
                         "(input_qubits * reupload_rounds must be >= n_step_features)")

    results.sort(key=lambda r: r["val_band_auprc"], reverse=True)
    csv_path = outdir / "tune_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    log(f"wrote {len(results)} trials -> {csv_path}")

    print("\n==== TOP CANDIDATES (validation-band AUPRC, single search seed) ====")
    print(f"  {'rank':<5}{'AUPRC':<9}{'qubits':<8}{'in':<4}{'R':<4}{'V':<4}"
          f"{'dt':<6}{'J':<6}{'h':<6}{'K':<4}")
    for i, r in enumerate(results[: args.top_k]):
        print(f"  {i + 1:<5}{r['val_band_auprc']:<9.4f}{r['n_qubits']:<8}"
              f"{r['input_qubits']:<4}{r['reupload_rounds']:<4}{r['virtual_nodes']:<4}"
              f"{r['dt']:<6}{r['J']:<6}{r['h']:<6}{r['K']:<4}")
    print(f"  {'--':<5}{baseline_auprc:<9.4f}  <- current config.yaml (reference)")
    print(f"  {'--':<5}{p0_auprc:<9.4f}  <- classical p0, no QRC at all (reference)")

    confirmed = confirm_top(exp, base_qrc, results[: args.top_k], cfg)
    json.dump(confirmed, open(outdir / "tune_top.json", "w"), indent=2)

    if not args.no_plots:
        write_tune_plots(outdir, results, confirmed, baseline_auprc, p0_auprc)

    best = confirmed[0]
    print("\n==== BEST CANDIDATE (per-seed, bootstrap CI vs classical p0) ====")
    print(json.dumps({k: best[k] for k in
                      ("n_qubits", "input_qubits", "reupload_rounds", "virtual_nodes",
                       "dt", "J", "h", "K", "val_band_auprc_mean",
                       "holds_across_all_seeds", "per_seed")}, indent=2))
    if best["holds_across_all_seeds"]:
        print("  -> beats classical p0 on validation with a 95% CI entirely > 0, "
              f"for every seed in {exp.seeds}.")
    else:
        print("  -> does NOT clear the p0 baseline with a significant CI for every "
              "seed on validation. Consider: widen --space, raise --trials, or "
              "accept that this entity key / feature set may be near the QRC's "
              "ceiling (see the causal-smoothing lever in report.md instead).")

    tuned_cfg = json.loads(json.dumps(cfg))  # deep copy
    for k in ("n_qubits", "input_qubits", "reupload_rounds", "virtual_nodes", "dt", "J", "h"):
        tuned_cfg["qrc"][k] = best[k]
    tuned_cfg["sequence"]["K"] = best["K"]
    tuned_cfg["run"]["outdir"] = "experiments/qrc_tuned"
    tuned_path = outdir / "tuned_config.yaml"
    yaml.safe_dump(tuned_cfg, open(tuned_path, "w"), sort_keys=False)
    print(f"\nWrote {tuned_path}")
    print("Confirm honestly on the frozen test split with:")
    print(f"  python run_experiment.py --config {tuned_path} --data {args.data} "
          f"--outdir experiments/ieee_uid_tuned")


if __name__ == "__main__":
    main()
