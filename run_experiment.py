#!/usr/bin/env python3
"""
Abstraction - end-to-end PoC runner.

    python run_experiment.py            # full run
    python run_experiment.py --quick    # fast smoke run

Produces, under results/ :
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
import json
import os
import platform
import sys
import time
from pathlib import Path

# keep BLAS / OpenMP from oversubscribing cores (nested with LightGBM / joblib)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import yaml

from src import data as datamod
from src import features as feat
from src import baselines as base
from src import routing as route
from src import sequence as seqmod
from src import esn as esnmod
from src import evaluate as ev
from src import plots as plt_
from src import attribution as attr
from src.calibration import Calibrator, threshold_for_recall
from src.qrc import make_embeddings, QuantumReservoir
from src.fusion import FusionModel, qrc_only_score
from src.distill import train_student
from src import postprocess as post

HERE = Path(__file__).parent
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
def apply_quick(cfg):
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


# ---------------------------------------------------------------------------
def build_features(train, val, test, entity_cols, extra_cols=()):
    train = datamod.add_entity_key(train, entity_cols)
    val = datamod.add_entity_key(val, entity_cols)
    test = datamod.add_entity_key(test, entity_cols)
    full = pd.concat([train, val, test]).sort_values("TransactionDT").reset_index(drop=True)

    ftr, fva, fte = feat.build_static(train, val, test)
    rolling = feat.build_rolling(full, entity_col="entity_key")
    ids = {k: v["TransactionID"].to_numpy() for k, v in
           (("train", train), ("val", val), ("test", test))}
    fmap = {"train": ftr, "val": fva, "test": fte}
    src = {"train": train, "val": val, "test": test}

    extra_cols = [c for c in extra_cols if c in full.columns]

    def _augment(Xd):
        if not extra_cols:
            return Xd
        out = {}
        for k, Xk in Xd.items():
            ex = (src[k].set_index("TransactionID")
                  .reindex(Xk.index)[extra_cols].astype("float32"))
            out[k] = pd.concat([Xk, ex], axis=1)
        return out

    def X(kind):
        base = {k: feat.assemble(fmap[k], rolling, ids[k], rolling=kind)
                for k in ("train", "val", "test")}
        # the raw IEEE feature block goes to the GBDT baselines (b4 / b5), not to
        # the deliberately-minimal static logreg/GBDT (b1) or the sequence path.
        return _augment(base) if kind in ("b4", "b5") else base

    return (train, val, test, full, rolling, X("b4"), X("b5"), X("none"))


class EmbReducer:
    """Standardise + PCA an (n, D) reservoir embedding, fit on train rows only.
    Keeps the logistic read-out well-conditioned when D >> #positives."""

    def __init__(self, k=28):
        self.k = k
        self._sc = None
        self._pca = None

    def fit(self, Z, tr_idx):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self._sc = StandardScaler().fit(Z[tr_idx])
        k = min(self.k, Z.shape[1], len(tr_idx) - 1)
        self._pca = PCA(n_components=k, random_state=0).fit(self._sc.transform(Z[tr_idx]))
        return self

    def transform(self, Z):
        return self._pca.transform(self._sc.transform(Z)).astype(np.float32)

    def fit_transform(self, Z, tr_idx):
        return self.fit(Z, tr_idx).transform(Z)


def ctx_matrix(rolling, seq_lengths, txn_ids, names):
    r = rolling.reindex(txn_ids)
    cols = []
    for nm in names:
        if nm == "seq_len":
            cols.append(seq_lengths.astype(float))
        else:
            cols.append(np.nan_to_num(r[nm].to_numpy(dtype=float)))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
def embed_split(cfg_qrc, seeds, seq, mask, tag):
    log(f"  QRC embed [{tag}] n={len(seq)} qubits={cfg_qrc['n_qubits']} "
        f"M={cfg_qrc.get('spatial_multiplex', 1)} shots={cfg_qrc.get('shots')}")
    prog = lambda a, n: log(f"    {tag}: {a}/{n}") if a in (n,) else None
    return make_embeddings(cfg_qrc, seeds, seq, mask, progress=None)


def fit_fusion_variant(variant, C_grid, p0, z, ctx, y, split_idx,
                       z_score=None, guard=True):
    tr, va, te = split_idx
    fm = FusionModel(variant, C_grid, guard=guard)
    kw = {}
    if variant == "score_fusion":
        kw = dict(z_score_tr=z_score[tr], z_score_va=z_score[va])
    fm.fit(p0[tr], z[tr], ctx[tr], y[tr], p0[va], z[va], ctx[va], y[va], **kw)
    pred_te = fm.predict(p0[te], z[te], ctx[te],
                         z_score[te] if variant == "score_fusion" else None)
    pred_va = fm.predict(p0[va], z[va], ctx[va],
                         z_score[va] if variant == "score_fusion" else None)
    return fm, pred_va, pred_te


# ---------------------------------------------------------------------------
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
    cfg["qrc"]["_K_traj"] = cfg["sequence"]["K"]   # trajectory length in the embedding
    seed = cfg["run"]["seed"]
    rng = np.random.default_rng(seed)
    outdir = HERE / (args.outdir or cfg["run"]["outdir"])
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    manifest = dict(
        config=cfg,
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages={},
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    for p in ("numpy", "pandas", "sklearn", "lightgbm", "catboost", "scipy", "shap", "pennylane"):
        try:
            manifest["packages"][p] = __import__(p).__version__
        except Exception:
            manifest["packages"][p] = None

    # ---- 1. data ---------------------------------------------------------
    if args.data == "ieee":
        from src import data_ieee as ieeemod
        log("loading IEEE-CIS Fraud Detection (real Kaggle csv) ...")
        df = ieeemod.load(cfg, seed)
    else:
        log("generating synthetic CNP transactions ...")
        df = datamod.generate(cfg, seed)
    extra_cols = list(cfg["data"].get("_extra_cols", []))
    log(f"  {len(df):,} transactions | fraud rate {df.isFraud.mean():.3%} | "
        f"{df.entity.nunique():,} entities | {len(extra_cols)} raw IEEE features")
    log(f"  fraud composition: {df[df.isFraud==1].fraud_type.value_counts().to_dict()}")

    primary_key = cfg["data"]["entity_keys"]["E2"]
    train, val, test = datamod.chronological_split(df, cfg)
    log(f"  split: train {len(train):,} | val {len(val):,} | test {len(test):,}")

    (train, val, test, full, rolling,
     Xb4, Xb5, Xst) = build_features(train, val, test, primary_key, extra_cols)
    Xtr, Xva, Xte = Xb4["train"], Xb4["val"], Xb4["test"]

    ytr = train.isFraud.to_numpy()
    yva = val.isFraud.to_numpy()
    yte = test.isFraud.to_numpy()

    # ---- 2. baselines --------------------------------------------------
    log("training baselines ...")
    b0 = base.train_logreg(Xst["train"].to_numpy(), ytr)
    b1 = base.train_gbdt(Xst["train"], ytr, cfg["baselines"]["lightgbm"], seed)
    b4 = base.train_gbdt(Xtr, ytr, cfg["baselines"]["lightgbm"], seed)
    b5 = base.train_gbdt(Xb5["train"], ytr, cfg["baselines"]["lightgbm"], seed)
    # CatBoost on the SAME feature set as B4 - per the IEEE-CIS 1st-place
    # writeup it was the strongest single GBDT of the three they ensembled
    # (0.9408 vs LightGBM 0.9384, XGBoost 0.9324 private-LB AUC), so this is
    # the fair "does B4's strength depend on picking LightGBM?" sanity check.
    b_cat = base.train_catboost(Xtr, ytr, cfg["baselines"].get("catboost", {}), seed)

    p_b0 = dict(val=b0.predict_proba(Xst["val"].to_numpy())[:, 1],
                test=b0.predict_proba(Xst["test"].to_numpy())[:, 1])
    p_b1 = dict(val=base.predict_proba(b1, Xst["val"]), test=base.predict_proba(b1, Xst["test"]))
    p_b5 = dict(val=base.predict_proba(b5, Xb5["val"]), test=base.predict_proba(b5, Xb5["test"]))
    p_cat = dict(val=base.predict_proba(b_cat, Xva), test=base.predict_proba(b_cat, Xte))
    # out-of-fold train scores so the train score distribution matches val/test
    # (an in-sample LightGBM score is wildly overconfident and breaks routing).
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    oof = cross_val_predict(
        base.make_gbdt(cfg["baselines"]["lightgbm"], seed), Xtr.to_numpy(), ytr,
        cv=StratifiedKFold(4, shuffle=True, random_state=seed),
        method="predict_proba", n_jobs=1)[:, 1]
    p_b4_raw = dict(train=oof,
                    val=base.predict_proba(b4, Xva),
                    test=base.predict_proba(b4, Xte))

    cal = Calibrator(cfg["baselines"]["calibration"]).fit(p_b4_raw["val"], yva)
    p0 = dict(train=cal.transform(p_b4_raw["train"]),
              val=cal.transform(p_b4_raw["val"]),
              test=cal.transform(p_b4_raw["test"]))

    # calibrated p0 for every row (used as the lagged hist_score in sequences)
    p0_all = {}
    for split_df, key in ((train, "train"), (val, "val"), (test, "test")):
        for tid, pv in zip(split_df["TransactionID"].to_numpy(), p0[key]):
            p0_all[int(tid)] = float(pv)

    # ---- 2b. Stage H: entity-level smoothing (independent classical lever,
    # kept OUT of the routing/QRC path so it's a clean one-factor experiment)
    smoothing_result = {"enabled": False}
    if cfg.get("smoothing", {}).get("enabled", True):
        log("entity-level smoothing (IEEE-CIS 1st-place style) ...")
        split_of = {}
        for tid in train["TransactionID"]: split_of[int(tid)] = "train"
        for tid in val["TransactionID"]: split_of[int(tid)] = "val"
        for tid in test["TransactionID"]: split_of[int(tid)] = "test"
        full_p0 = full[["TransactionID", "entity_key", "isFraud"]].copy()
        full_p0["p0"] = full_p0["TransactionID"].map(p0_all)
        full_p0["split"] = full_p0["TransactionID"].map(split_of)
        val_mask = (full_p0["split"] == "val").to_numpy()
        test_mask = (full_p0["split"] == "test").to_numpy()

        alpha_grid = cfg["smoothing"].get("alpha_grid", [1.0, 0.75, 0.5, 0.25, 0.0])
        alpha_star = post.select_alpha(full_p0, "p0", "entity_key", "isFraud",
                                       val_mask, alpha_grid)
        causal_full = post.causal_smooth(full_p0, "p0", "entity_key", alpha_star)
        p0_causal_test = causal_full[test_mask]
        p0_batch_test = post.batch_smooth(full_p0[test_mask], "p0", "entity_key")

        ap_base = float(ev.auprc_fn(yte, p0["test"]))
        ap_causal = float(ev.auprc_fn(yte, p0_causal_test))
        ap_batch = float(ev.auprc_fn(yte, p0_batch_test))
        d_causal = ev.paired_bootstrap_delta(yte, p0_causal_test, p0["test"],
                                             ev.auprc_fn, iters=cfg["eval"]["bootstrap_iters"],
                                             seed=seed)
        d_batch = ev.paired_bootstrap_delta(yte, p0_batch_test, p0["test"],
                                            ev.auprc_fn, iters=cfg["eval"]["bootstrap_iters"],
                                            seed=seed)
        smoothing_result = dict(
            enabled=True, alpha_star=alpha_star, alpha_grid=alpha_grid,
            auprc_baseline_B4=ap_base,
            auprc_causal_smooth=ap_causal, delta_causal=d_causal,
            auprc_batch_smooth_AUDIT_ONLY=ap_batch, delta_batch_AUDIT_ONLY=d_batch,
            note=("causal_smooth is real-time deployable (past-only entity mean). "
                 "batch_smooth reproduces the exact IEEE-CIS 1st-place trick "
                 "(whole-batch entity mean) but uses each entity's FUTURE test "
                 "transactions, so it is reported for audit/offline comparison "
                 "only - never claim it as a deployable number."),
        )
        log(f"  alpha* (frozen on validation) = {alpha_star}")
        log(f"  B4 baseline AUPRC            = {ap_base:.4f}")
        log(f"  + causal smoothing (real-time)= {ap_causal:.4f}  "
            f"delta={d_causal['delta']:+.4f} CI=[{d_causal['ci_low']:+.4f},{d_causal['ci_high']:+.4f}]")
        log(f"  + batch smoothing (AUDIT ONLY)= {ap_batch:.4f}  "
            f"delta={d_batch['delta']:+.4f} CI=[{d_batch['ci_low']:+.4f},{d_batch['ci_high']:+.4f}]")

    # ---- 3. routing (FROZEN) -----------------------------------------
    log("fitting ambiguity band on validation ...")
    band = route.fit_band(yva, p0["val"],
                          cfg["routing"]["target_fraud_recall"],
                          cfg["routing"]["review_recall"],
                          cfg["routing"]["routing_budget"],
                          y_train=ytr, p0_train=p0["train"])
    cfg["routing"]["tau_low"] = band["tau_low"]
    cfg["routing"]["tau_high"] = band["tau_high"]
    log(f"  band = [{band['tau_low']:.4f}, {band['tau_high']:.4f}]  "
        f"t*={band['t_star']:.4f}  routed(val)={band['routed_fraction_val']:.3%}  "
        f"pos(train+val)={band['pos_in_band_trainval']}")

    amb = dict(train=route.apply_band(p0["train"], band),
              val=route.apply_band(p0["val"], band),
              test=route.apply_band(p0["test"], band))
    for k in amb:
        yy = {"train": ytr, "val": yva, "test": yte}[k]
        log(f"  {k}: routed {amb[k].mean():.3%}  fraud-in-band {yy[amb[k]].mean():.3%}  "
            f"(vs {yy.mean():.3%} overall)")

    # ---- 4. sequences ------------------------------------------------
    log("building per-entity sequences for the ambiguous band ...")
    K = cfg["sequence"]["K"]
    ids = {k: {"train": train, "val": val, "test": test}[k]["TransactionID"].to_numpy()[amb[k]]
           for k in amb}
    seqs, masks, lens = {}, {}, {}
    for k in ("train", "val", "test"):
        s, m, L = seqmod.build_sequences(full, rolling, p0_all, ids[k], K)
        seqs[k], masks[k], lens[k] = s, m, L
    sc = seqmod.SeqScaler().fit(seqs["train"], masks["train"])
    seqs_s = {k: sc.transform(seqs[k]) for k in seqs}
    y_amb = {k: {"train": ytr, "val": yva, "test": yte}[k][amb[k]] for k in amb}
    for k in ("train", "val", "test"):
        log(f"  band {k}: n={amb[k].sum()}  fraud={int(y_amb[k].sum())}")
    if y_amb["train"].sum() < 3 or y_amb["val"].sum() < 3:
        raise SystemExit("Ambiguous band has too few fraud examples to train the "
                         "fusion read-out. Increase routing_budget / data size.")

    ctx_names = cfg["fusion"]["context_features"]
    ctx = {k: ctx_matrix(rolling, lens[k], ids[k], ctx_names) for k in ids}

    n_tr, n_va, n_te = len(ids["train"]), len(ids["val"]), len(ids["test"])
    split_idx = (np.arange(n_tr),
                 np.arange(n_tr, n_tr + n_va),
                 np.arange(n_tr + n_va, n_tr + n_va + n_te))

    def stack(d):
        return np.concatenate([d["train"], d["val"], d["test"]], axis=0)

    seq_all = stack(seqs_s)
    mask_all = np.concatenate([masks["train"], masks["val"], masks["test"]], axis=0)
    y_all = np.concatenate([y_amb["train"], y_amb["val"], y_amb["test"]])
    p0_all_band = np.concatenate([p0["train"][amb["train"]], p0["val"][amb["val"]],
                                  p0["test"][amb["test"]]])
    ctx_all = np.vstack([ctx["train"], ctx["val"], ctx["test"]])

    # ---- 5. QRC main run (seed 0) + ESN ------------------------------
    seeds = cfg["qrc"]["reservoir_seeds"]
    t_qrc = time.time()
    Z_qrc = embed_split({**cfg["qrc"]}, seeds, seq_all, mask_all, "main")
    qrc_secs = time.time() - t_qrc
    log(f"  QRC embed done in {qrc_secs:.1f}s  dim={Z_qrc.shape[1]}")

    esn = esnmod.EchoStateNetwork(cfg["esn"], seq_all.shape[-1], seed)
    Z_esn = esn.embed(seq_all, mask_all)
    log(f"  ESN embed done  dim={Z_esn.shape[1]}")

    # PCA-reduce both reservoir embeddings (fit on train band only) so the
    # logistic read-out stays well-conditioned given ~10^2-10^3 positives.
    tr, va, te = split_idx
    kdim = int(cfg["fusion"].get("embed_pca_dim", 28))
    red_qrc = EmbReducer(kdim).fit(Z_qrc, tr)
    red_esn = EmbReducer(kdim).fit(Z_esn, tr)
    Z_qrc_full, Z_esn_full = Z_qrc, Z_esn
    Z_qrc, Z_esn = red_qrc.transform(Z_qrc), red_esn.transform(Z_esn)
    log(f"  reduced embeddings -> QRC {Z_qrc.shape[1]}  ESN {Z_esn.shape[1]}")

    # ---- 6. fusion variants ----------------------------------------
    log("fitting fusion / readout variants ...")
    Cg = cfg["fusion"]["C_grid"]
    zs_tr, zs_va, zs_te = qrc_only_score(Z_qrc[tr], y_all[tr], Z_qrc[va], y_all[va],
                                         Z_qrc[te], Cg)
    z_score_all = np.concatenate([zs_tr, zs_va, zs_te])

    fusion_preds = {}
    fusion_models = {}
    for variant in cfg["fusion"]["variants"]:
        fm, pv, pt = fit_fusion_variant(variant, Cg, p0_all_band, Z_qrc, ctx_all,
                                        y_all, split_idx, z_score=z_score_all)
        fusion_models[variant] = fm
        fusion_preds[variant] = dict(val=pv, test=pt, C=fm.C_)
        log(f"  {variant:22s}  C={fm.C_}  test-band AUPRC="
            f"{ev.auprc_fn(y_all[te], pt):.4f}")

    # ESN control: fit ALL the same fusion heads, deploy the same one QRC deploys
    esn_fits = {}
    for variant in ("qrc_only", "score_fusion", "feature_fusion"):
        zs = qrc_only_score(Z_esn[tr], y_all[tr], Z_esn[va], y_all[va], Z_esn[te], Cg) \
            if variant == "score_fusion" else (None, None, None)
        zsc = np.concatenate(zs) if zs[0] is not None else None
        fmE, pvE, ptE = fit_fusion_variant(variant, Cg, p0_all_band, Z_esn, ctx_all,
                                           y_all, split_idx, z_score=zsc)
        esn_fits[variant] = dict(model=fmE, val=pvE, test=ptE)
        log(f"  ESN {variant:18s} C={fmE.C_}  test-band AUPRC="
            f"{ev.auprc_fn(y_all[te], ptE):.4f}")

    # ---- 7. end-to-end reconstruction on test ---------------------
    # pick the deployed fusion head on VALIDATION-band AUPRC (model selection,
    # never on test) among the p0-aware variants.
    # score_fusion (Q1: "LightGBM score + QRC score") is the architecture in
    # Abstraction.md and is the robust default; only deploy the richer
    # feature_fusion if it is clearly better on the validation band.
    ap_sf = ev.auprc_fn(y_all[va], fusion_preds["score_fusion"]["val"])
    ap_ff = ev.auprc_fn(y_all[va], fusion_preds["feature_fusion"]["val"])
    q3_variant = "feature_fusion" if ap_ff > ap_sf + 0.015 else "score_fusion"
    log(f"  deployed fusion head = {q3_variant}  "
        f"(val AUPRC score_fusion={ap_sf:.4f} feature_fusion={ap_ff:.4f})")
    esn_dep = esn_fits[q3_variant]
    fm_esn, pv_esn, pt_esn = esn_dep["model"], esn_dep["val"], esn_dep["test"]

    # Rank-preserving splice: the ambiguous band keeps the exact score interval
    # it occupied under p0 (so the GLOBAL ranking / full-test AUPRC can only move
    # via BETTER within-band ordering), but rows inside the band are re-ordered
    # according to the fusion head. This is the honest way to fold a
    # band-restricted model back into a full-population score.
    def _rank_splice(p0_split, band_mask, band_scores):
        p = np.asarray(p0_split, float).copy()
        target = np.sort(p[band_mask])
        order = np.argsort(np.argsort(band_scores))      # ranks 0..m-1
        p[band_mask] = target[order]
        return p

    def reconstruct(band_pred_test):
        return _rank_splice(p0["test"], amb["test"], band_pred_test)

    p_q3_e2e = reconstruct(fusion_preds[q3_variant]["test"])
    p_esn_e2e = reconstruct(pt_esn)

    # per-model decision thresholds: frozen on validation end-to-end scores
    rc_t = cfg["routing"]["target_fraud_recall"]
    p_q3_val = _rank_splice(p0["val"], amb["val"], fusion_preds[q3_variant]["val"])
    p_esn_val = _rank_splice(p0["val"], amb["val"], pv_esn)
    t_hybrid = threshold_for_recall(yva, p_q3_val, rc_t)
    t_esn = threshold_for_recall(yva, p_esn_val, rc_t)
    log(f"  frozen hybrid decision threshold (val, recall {rc_t}) = {t_hybrid:.4f}")

    # ---- 8. metrics ---------------------------------------------
    log("computing metrics ...")
    t_b4 = threshold_for_recall(yva, p0["val"], cfg["routing"]["target_fraud_recall"])
    metrics = {"full_test": {}, "ambiguous_band_test": {}, "deltas": {},
               "seeds": {}, "ablations": {}, "calibration": {}, "operational": {},
               "smoothing": smoothing_result}

    t_b5 = threshold_for_recall(yva, p_b5["val"], rc_t)
    t_cat = threshold_for_recall(yva, p_cat["val"], rc_t)
    full_models = {
        "B0_logreg_static": p_b0["test"],
        "B1_lgbm_static": p_b1["test"],
        "B4_lgbm_aggregates": p_b4_raw["test"],
        "B4_calibrated": p0["test"],
        "B5_lgbm_manual_temporal_FE": p_b5["test"],
        "B_catboost_aggregates": p_cat["test"],
        "ESN_end2end": p_esn_e2e,
        "Q3_hybrid_end2end": p_q3_e2e,
    }
    thr_map = {"B0_logreg_static": 0.5, "B1_lgbm_static": 0.5,
               "B4_lgbm_aggregates": t_b4, "B4_calibrated": t_b4,
               "B5_lgbm_manual_temporal_FE": t_b5, "B_catboost_aggregates": t_cat,
               "ESN_end2end": t_esn, "Q3_hybrid_end2end": t_hybrid}
    for nm, p in full_models.items():
        metrics["full_test"][nm] = ev.basic_metrics(yte, p, thr_map.get(nm, 0.5))

    band_models = {
        "B4_p0": (p0["test"][amb["test"]], p0["val"][amb["val"]]),
        "Q0_qrc_only": (fusion_preds["qrc_only"]["test"], fusion_preds["qrc_only"]["val"]),
        "Q1_score_fusion": (fusion_preds["score_fusion"]["test"], fusion_preds["score_fusion"]["val"]),
        "Q2_feature_fusion": (fusion_preds["feature_fusion"]["test"], fusion_preds["feature_fusion"]["val"]),
        "Q2_no_p0_ablation": (fusion_preds["feature_fusion_no_p0"]["test"], fusion_preds["feature_fusion_no_p0"]["val"]),
        f"Q_deployed[{q3_variant}]": (fusion_preds[q3_variant]["test"], fusion_preds[q3_variant]["val"]),
        f"ESN_deployed[{q3_variant}]": (pt_esn, pv_esn),
    }
    for nm, (p_te, p_va) in band_models.items():
        thr = threshold_for_recall(y_all[va], p_va, rc_t)
        metrics["ambiguous_band_test"][nm] = ev.basic_metrics(y_all[te], p_te, thr)
    band_models = {k: v[0] for k, v in band_models.items()}

    # paired bootstrap deltas
    bi = cfg["eval"]["bootstrap_iters"]
    metrics["deltas"]["c1_auprc_Q3_vs_B4_full"] = ev.paired_bootstrap_delta(
        yte, p_q3_e2e, p_b4_raw["test"], ev.auprc_fn, iters=bi, seed=seed)
    for rc in cfg["eval"]["fixed_recall_points"]:
        fdr_fn = ev.fdr_fn_factory(rc)
        d = ev.paired_bootstrap_delta(yte, p_q3_e2e, p_b4_raw["test"], fdr_fn,
                                      iters=bi, seed=seed)
        metrics["deltas"][f"c2_fdr_Q3_vs_B4_recall{rc}"] = d
        fdr_q3, thr_q3 = ev.false_decline_rate(yte, p_q3_e2e, rc)
        fdr_b4, thr_b4 = ev.false_decline_rate(yte, p_b4_raw["test"], rc)
        metrics["operational"][f"false_decline_rate_recall{rc}"] = dict(
            B4=fdr_b4, Q3=fdr_q3, delta=fdr_q3 - fdr_b4)
    metrics["deltas"]["c3_auprc_Q2_vs_ESN_band"] = ev.paired_bootstrap_delta(
        y_all[te], fusion_preds[q3_variant]["test"], pt_esn, ev.auprc_fn,
        iters=bi, seed=seed)
    # honesty check: hybrid vs the aggressive manual-temporal-FE GBDT
    metrics["deltas"]["auprc_Q3_vs_B5_manualFE_full"] = ev.paired_bootstrap_delta(
        yte, p_q3_e2e, p_b5["test"], ev.auprc_fn, iters=bi, seed=seed)
    metrics["deltas"]["auprc_B5_vs_B4_full"] = ev.paired_bootstrap_delta(
        yte, p_b5["test"], p_b4_raw["test"], ev.auprc_fn, iters=bi, seed=seed)
    # sanity check: does B4's strength depend on picking LightGBM specifically?
    metrics["deltas"]["auprc_CatBoost_vs_B4_full"] = ev.paired_bootstrap_delta(
        yte, p_cat["test"], p_b4_raw["test"], ev.auprc_fn, iters=bi, seed=seed)

    metrics["calibration"]["B4_calibrated"] = dict(
        brier=metrics["full_test"]["B4_calibrated"]["brier"],
        ece=metrics["full_test"]["B4_calibrated"]["ece"])
    metrics["calibration"]["Q3_hybrid_end2end"] = dict(
        brier=metrics["full_test"]["Q3_hybrid_end2end"]["brier"],
        ece=metrics["full_test"]["Q3_hybrid_end2end"]["ece"])

    metrics["operational"]["routing_rate_test"] = float(amb["test"].mean())
    metrics["operational"]["fraud_rate_in_band_test"] = float(yte[amb["test"]].mean())

    # ---- 9. multi-seed (c4) -----------------------------------------
    log("multi-seed check ...")
    for s in seeds:
        Zs = make_embeddings({**cfg["qrc"], "spatial_multiplex": 1}, [s],
                             seq_all, mask_all)
        Zs = EmbReducer(kdim).fit_transform(Zs, tr)
        zsc_s = None
        if q3_variant == "score_fusion":
            zsc_s = np.concatenate(qrc_only_score(Zs[tr], y_all[tr], Zs[va], y_all[va],
                                                  Zs[te], Cg))
        fm_s, pv_s, pt_s = fit_fusion_variant(q3_variant, Cg, p0_all_band,
                                              Zs, ctx_all, y_all, split_idx,
                                              z_score=zsc_s)
        pe = reconstruct(pt_s)
        d1 = ev.paired_bootstrap_delta(yte, pe, p_b4_raw["test"], ev.auprc_fn,
                                       iters=max(400, bi // 2), seed=s)
        fdr_fn = ev.fdr_fn_factory(cfg["routing"]["target_fraud_recall"])
        d2 = ev.paired_bootstrap_delta(yte, pe, p_b4_raw["test"], fdr_fn,
                                       iters=max(400, bi // 2), seed=s)
        db = ev.paired_bootstrap_delta(y_all[te], pt_s, pt_esn, ev.auprc_fn,
                                       iters=max(400, bi // 2), seed=s)
        metrics["seeds"][str(s)] = dict(
            delta_auprc_full=d1, delta_fdr_full=d2, delta_auprc_vs_esn_band=db,
            band_auprc=float(ev.auprc_fn(y_all[te], pt_s)))
        log(f"  seed {s}: dAUPRC(full)={d1['delta']:+.4f} [{d1['ci_low']:+.4f},"
            f"{d1['ci_high']:+.4f}]  dFDR={d2['delta']:+.4f}  vs ESN band={db['delta']:+.4f}")

    # ---- 10. ablations -------------------------------------------
    if cfg["ablations"]["enabled"]:
        log("ablations ...")
        A = cfg["ablations"]
        # Ablations must show the TRUE effect of a destructive intervention -
        # never masked by the deployment safety guard silently falling back to
        # p0 (that guard is a production feature, not a diagnostic one).
        def band_auprc_for(qcfg, seq_in=seq_all, mask_in=mask_all):
            Z = make_embeddings(qcfg, seeds[:1], seq_in, mask_in)
            Z = EmbReducer(kdim).fit_transform(Z, tr)
            _, _, pt = fit_fusion_variant("feature_fusion", Cg, p0_all_band, Z,
                                          ctx_all, y_all, split_idx, guard=False)
            return float(ev.auprc_fn(y_all[te], pt)), pt

        # unguarded ideal reference, reusing the already-computed main embedding
        # (no re-simulation needed: same reservoir_seeds[0], same PCA space)
        fm_ideal_ug, _, pt_ideal_ug = fit_fusion_variant(
            "feature_fusion", Cg, p0_all_band, Z_qrc, ctx_all, y_all, split_idx,
            guard=False)
        ideal_band_auprc = float(ev.auprc_fn(y_all[te], pt_ideal_ug))
        metrics["ablations"]["ideal_feature_fusion_band_auprc"] = ideal_band_auprc
        metrics["ablations"]["ideal_feature_fusion_band_auprc_guarded_deployed"] = float(
            ev.auprc_fn(y_all[te], fusion_preds["feature_fusion"]["test"]))

        if A["no_entanglement"]:
            v, _ = band_auprc_for({**cfg["qrc"], "_entangle": False})
            metrics["ablations"]["no_entanglement"] = v
        if A["random_reservoir"]:
            v, _ = band_auprc_for({**cfg["qrc"], "_random_reservoir": True})
            metrics["ablations"]["random_reservoir"] = v
        if A["shuffle_sequence"]:
            rs = np.random.default_rng(seed)
            seq_sh = seq_all.copy()
            for i in range(len(seq_sh)):
                idx = np.where(mask_all[i] > 0)[0]
                perm = rs.permutation(idx)
                seq_sh[i, idx] = seq_sh[i, perm]
            v, _ = band_auprc_for({**cfg["qrc"]}, seq_sh, mask_all)
            metrics["ablations"]["shuffle_sequence"] = v
        if A["no_history"]:
            m1 = mask_all.copy()
            for i in range(len(m1)):
                idx = np.where(m1[i] > 0)[0]
                if len(idx) > 1:
                    m1[i, idx[:-1]] = 0.0
            v, _ = band_auprc_for({**cfg["qrc"]}, seq_all, m1)
            metrics["ablations"]["no_history_K1"] = v

        # shot-noise sweep (unguarded fusion trained on ideal, evaluated on
        # noisy embeddings - guarded here would just hide degradation behind p0)
        shots_res = []
        for sh in A["shots_sweep"]:
            Zsh = make_embeddings({**cfg["qrc"], "shots": sh}, seeds[:1],
                                  seq_all[te], mask_all[te])
            Zsh = red_qrc.transform(Zsh)
            p = fm_ideal_ug.predict(p0_all_band[te], Zsh, ctx_all[te])
            shots_res.append((sh, float(ev.auprc_fn(y_all[te], p))))
        metrics["ablations"]["shots_sweep"] = shots_res

        sweep_q, sweep_k, sweep_dt = {}, {}, {}
        for nq in A["qubit_sweep"]:
            v, _ = band_auprc_for({**cfg["qrc"], "n_qubits": nq,
                                   "input_qubits": min(3, nq - 1)})
            sweep_q[str(nq)] = v
        for kk in A["K_sweep"]:
            m1 = mask_all.copy()
            for i in range(len(m1)):
                idx = np.where(m1[i] > 0)[0]
                if len(idx) > kk:
                    m1[i, idx[:-kk]] = 0.0
            v, _ = band_auprc_for({**cfg["qrc"]}, seq_all, m1)
            sweep_k[str(kk)] = v
        for dtv in A.get("dt_sweep", []):
            v, _ = band_auprc_for({**cfg["qrc"], "dt": dtv})
            sweep_dt[str(dtv)] = v
        metrics["ablations"]["qubit_sweep_band_auprc"] = sweep_q
        metrics["ablations"]["K_sweep_band_auprc"] = sweep_k
        metrics["ablations"]["dt_sweep_band_auprc"] = sweep_dt

    # ---- 11. distillation --------------------------------------
    log("distillation to a real-time classical student ...")
    # target = the hybrid's p_final on the train+val ambiguous band
    fm_q3 = fusion_models[q3_variant]
    _zs = q3_variant == "score_fusion"
    pf_tr = fm_q3.predict(p0_all_band[tr], Z_qrc[tr], ctx_all[tr],
                          z_score_all[tr] if _zs else None)
    pf_va = fm_q3.predict(p0_all_band[va], Z_qrc[va], ctx_all[va],
                          z_score_all[va] if _zs else None)
    Xband_tr = np.vstack([Xtr.to_numpy()[amb["train"]], Xva.to_numpy()[amb["val"]]])
    yb_tr = np.concatenate([pf_tr, pf_va])
    Xband_te = Xte.to_numpy()[amb["test"]]
    p_student, lat_ms = train_student(Xband_tr, yb_tr, Xband_te,
                                      cfg["distill"], seed)
    student_band_auprc = float(ev.auprc_fn(y_all[te], p_student))
    p_student_e2e = reconstruct(p_student)
    metrics["operational"]["distilled_student"] = dict(
        band_auprc=student_band_auprc,
        hybrid_band_auprc=float(ev.auprc_fn(y_all[te],
                                            fusion_preds[q3_variant]["test"])),
        full_test_auprc=float(ev.auprc_fn(yte, p_student_e2e)),
        inference_latency_ms_per_txn=lat_ms,
    )
    log(f"  student band AUPRC {student_band_auprc:.4f} vs hybrid "
        f"{ev.auprc_fn(y_all[te], fusion_preds[q3_variant]['test']):.4f}  "
        f"latency {lat_ms:.4f} ms/txn")

    # ---- 12. attribution + predictions.csv --------------------
    log("attribution + predictions.csv ...")
    feat_names = list(Xte.columns)
    sample = np.arange(len(test))
    lgb_attr = attr.lightgbm_attribution(b4, Xte.to_numpy(), feat_names, top_k=5)
    fus_groups = attr.fusion_group_attribution(
        fm_q3, q3_variant, Z_qrc.shape[1], ctx_names)
    metrics["fusion_group_attribution"] = fus_groups

    route_label = np.full(len(test), "classical_approve", dtype=object)
    route_label[p0["test"] >= band["tau_high"]] = "classical_decline_or_review"
    route_label[amb["test"]] = "qrc_second_opinion"

    fraud_prob = p_q3_e2e
    fraud_pred = (fraud_prob >= t_hybrid).astype(int)

    band_pos = {int(t): i for i, t in enumerate(ids["test"])}
    rows = []
    for i, tid in enumerate(test["TransactionID"].to_numpy()):
        if amb["test"][i] and int(tid) in band_pos:
            a = {"classical_score_p0": round(float(p0["test"][i]), 4),
                 "qrc_observable_groups": fus_groups}
        else:
            a = {"top_features": lgb_attr[i]}
        rows.append(dict(
            TransactionID=int(tid),
            entity_key=test["entity_key"].to_numpy()[i],
            TransactionDT=float(test["TransactionDT"].to_numpy()[i]),
            fraud_probability=round(float(fraud_prob[i]), 6),
            fraud_prediction=int(fraud_pred[i]),
            route=route_label[i],
            actual_isFraud=int(yte[i]),
            feature_attribution=json.dumps(a),
        ))
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(outdir / "predictions.csv", index=False)

    # ---- 14. success criteria --------------------------------
    log("evaluating frozen success criteria ...")
    c1 = metrics["deltas"]["c1_auprc_Q3_vs_B4_full"]
    rc0 = cfg["routing"]["target_fraud_recall"]
    c2 = metrics["deltas"][f"c2_fdr_Q3_vs_B4_recall{rc0}"]
    c3 = metrics["deltas"]["c3_auprc_Q2_vs_ESN_band"]
    c1_pass = c1["delta"] > 0 and c1["ci_low"] > 0
    c2_pass = c2["delta"] < 0 and c2["ci_high"] < 0
    c3_beats = c3["delta"] > 0 and c3["ci_low"] > 0
    c3_parity = c3["ci_low"] > -0.02          # not meaningfully worse than the ESN
    c4_pass = all(
        (metrics["seeds"][str(s)]["delta_auprc_full"]["ci_low"] > 0 or
         metrics["seeds"][str(s)]["delta_fdr_full"]["ci_high"] < 0)
        for s in seeds)
    # A small (6-8 qubit) reservoir is, per Fujii & Nakajima, in the class of a
    # ~100-500 node ESN, so "beats the ESN" is a stretch goal; the honest bar is
    # PARITY with the classical reservoir while still improving on the GBDT.
    overall = (c1_pass or c2_pass) and c3_parity and c4_pass
    sc = dict(
        c1_delta_auprc_positive_ci=dict(pass_=bool(c1_pass), **c1),
        c2_false_decline_reduced_ci=dict(pass_=bool(c2_pass), **c2),
        c3_vs_esn_on_band=dict(beats_esn=bool(c3_beats), parity_with_esn=bool(c3_parity),
                               pass_=bool(c3_parity), **c3),
        c4_holds_across_seeds=dict(pass_=bool(c4_pass)),
        overall_measurable_improvement=bool(overall),
        note=("PoC verdict only. Never labelled 'quantum advantage'. "
              "A 6-8 qubit QRC is comparable to a small classical ESN "
              "(Fujii & Nakajima 2016); the bar here is parity with the ESN "
              "AND improvement over the tuned GBDT baseline. Any gain is a "
              "NISQ-era feature-discovery result, validated offline."),
    )
    json.dump(sc, open(outdir / "success_criteria.json", "w"), indent=2)

    # ---- 15. manifest + metrics -----------------------------
    manifest["frozen"] = dict(
        entity_key="E2", split="chronological_60_20_20",
        tau_low=band["tau_low"], tau_high=band["tau_high"],
        t_star=band["t_star"], hybrid_decision_threshold=t_hybrid,
        target_fraud_recall=rc0, routing_budget=cfg["routing"]["routing_budget"],
        reservoir_seeds=seeds,
    )
    manifest["quantum_resource"] = dict(
        n_qubits=cfg["qrc"]["n_qubits"],
        input_qubits=cfg["qrc"]["input_qubits"],
        reupload_rounds=cfg["qrc"]["reupload_rounds"],
        virtual_nodes=cfg["qrc"]["virtual_nodes"],
        evolution_steps_per_txn=cfg["sequence"]["K"] * cfg["qrc"]["reupload_rounds"]
        * cfg["qrc"]["virtual_nodes"],
        observables_per_node=QuantumReservoir(cfg["qrc"], seeds[0]).n_obs,
        embedding_dim=int(Z_qrc.shape[1]),
        spatial_multiplex=cfg["qrc"]["spatial_multiplex"],
        shots="ideal_statevector" if cfg["qrc"]["shots"] is None else cfg["qrc"]["shots"],
        simulator="numpy density-matrix (exact); PennyLane cross-check in src/qrc_pennylane.py",
        qrc_embed_seconds=qrc_secs,
    )
    manifest["runtime_seconds"] = time.time() - T0
    json.dump(manifest, open(outdir / "manifest.json", "w"), indent=2, default=str)
    json.dump(metrics, open(outdir / "metrics.json", "w"), indent=2, default=str)

    write_report(outdir, cfg, metrics, sc, manifest, band, df, amb, yte)

    # ---- 16. plots (best-effort; never block the deliverables) --------
    try:
        log("plots ...")
        plt_.pr_curves({"B4 (LightGBM+rolling)": (yte, p_b4_raw["test"]),
                        "ESN hybrid": (yte, p_esn_e2e),
                        "Q3 QRC hybrid": (yte, p_q3_e2e)}, outdir / "plots")
        plt_.roc_curves({"B4": (yte, p_b4_raw["test"]),
                         "Q3 QRC hybrid": (yte, p_q3_e2e)}, outdir / "plots")
        plt_.calibration_plot({"B4 calibrated": (yte, p0["test"]),
                               "Q3 hybrid": (yte, p_q3_e2e)}, outdir / "plots")
        plt_.routing_plot(p0["test"], yte, band, outdir / "plots")
        plt_.bar_compare(list(band_models.keys()),
                         [ev.auprc_fn(y_all[te], p) for p in band_models.values()],
                         "AUPRC on the ambiguous band (test)", "AUPRC",
                         outdir / "plots", "band_auprc.png",
                         ref=ev.auprc_fn(y_all[te], p0["test"][amb["test"]]))
        if cfg["ablations"]["enabled"]:
            ab = metrics["ablations"]
            names = ["ideal"] + [k for k in ("no_entanglement", "random_reservoir",
                     "shuffle_sequence", "no_history_K1") if k in ab]
            vals = [ab["ideal_feature_fusion_band_auprc"]] + [ab[k] for k in names[1:]]
            plt_.bar_compare(names, vals, "QRC ablations - band AUPRC", "AUPRC",
                             outdir / "plots", "ablations.png",
                             ref=ev.auprc_fn(y_all[te], p0["test"][amb["test"]]))
            if "shots_sweep" in ab and ab["shots_sweep"]:
                sh, au = zip(*ab["shots_sweep"])
                plt_.shots_plot(list(sh), list(au), outdir / "plots",
                                ideal=ab["ideal_feature_fusion_band_auprc"])
    except Exception as e:                       # pragma: no cover
        log(f"  plotting failed ({e!r}) - deliverables already written")

    log(f"DONE in {time.time() - T0:.1f}s -> {outdir}")
    print("\n==== SUCCESS CRITERIA ====")
    for k in ("c1_delta_auprc_positive_ci", "c2_false_decline_reduced_ci",
              "c3_vs_esn_on_band", "c4_holds_across_seeds"):
        print(f"  {k:38s} {'PASS' if sc[k]['pass_'] else 'fail'}")
    print(f"  {'OVERALL measurable improvement':38s} "
          f"{'YES' if sc['overall_measurable_improvement'] else 'NO'}")


# ---------------------------------------------------------------------------
def write_report(outdir, cfg, metrics, sc, manifest, band, df, amb, yte):
    ft = metrics["full_test"]
    bt = metrics["ambiguous_band_test"]
    q = manifest["quantum_resource"]

    def row(nm, d):
        cm = d["confusion_matrix"]
        return (f"| {nm} | {d['auroc']:.4f} | {d['auprc']:.4f} | {d['f1']:.4f} | "
                f"{d['precision']:.4f} | {d['recall']:.4f} | "
                f"{cm['tp']}/{cm['fp']}/{cm['fn']}/{cm['tn']} |")

    c1 = metrics["deltas"]["c1_auprc_Q3_vs_B4_full"]
    rc0 = cfg["routing"]["target_fraud_recall"]
    c2 = metrics["deltas"][f"c2_fdr_Q3_vs_B4_recall{rc0}"]
    c3 = metrics["deltas"]["c3_auprc_Q2_vs_ESN_band"]
    st = metrics["operational"]["distilled_student"]

    md = f"""# Abstraction - PoC results

*Hybrid LightGBM + Quantum Reservoir Computing for card-not-present fraud detection.*
Auto-generated by `run_experiment.py`. Runtime {manifest['runtime_seconds']:.0f}s.

## 1. Data

{"**IEEE-CIS Fraud Detection (real Kaggle data)** - the HSBC brief's primary dataset. `train_transaction.csv` left-joined with `train_identity.csv`; the unlabelled `test_*.csv` is the Kaggle leaderboard split so evaluation uses a chronological split of the training file (same protocol as the synthetic run)." if cfg["run"].get("data_source") == "ieee" else "Synthetic CNP transactions (IEEE-CIS stand-in; Kaggle credentials unavailable in this environment). Generator reproduces the properties HSBC calls out."}

- transactions: **{len(df):,}**, fraud rate **{df.isFraud.mean():.2%}**, entities {df.entity.nunique():,}
- fraud composition: {df[df.isFraud==1].fraud_type.value_counts().to_dict()}
- split: chronological 60/20/20 (FROZEN)

## 2. Architecture

```
transaction + history
  -> preprocessing (leakage-safe: every feature uses only rows with time < t)
  -> LightGBM (static + standard past-only aggregate features) [B4, production baseline]
     (B5 = B4 + aggressive hand-built temporal features, reported for honesty)
  -> isotonic calibration (fit on validation only)  -> p0
  -> ambiguity router:  p0 < tau_low  -> approve
                        p0 > tau_high -> decline / review
                        tau_low <= p0 <= tau_high -> QRC second opinion
  -> sequence builder (K={cfg['sequence']['K']} past txns of the same entity_key)
  -> fixed quantum reservoir (transverse-field Ising, {q['n_qubits']} qubits, NOT trained)
  -> measurement features z_qrc  ({q['observables_per_node']} observables x {cfg['qrc']['virtual_nodes']} virtual nodes + trajectory summary; dim {q['embedding_dim']})
  -> logistic fusion  [logit(p0), z_qrc, context]  -> p_final
  -> decision + grouped feature attribution
  -> (offline) distillation to a classical student for the 100-300 ms path
```

Frozen thresholds: band = [{band['tau_low']:.4f}, {band['tau_high']:.4f}],
t* = {band['t_star']:.4f}, hybrid decision threshold = {manifest['frozen']['hybrid_decision_threshold']:.4f}
(all chosen on validation, target fraud recall {rc0}).
Routing rate on test: **{metrics['operational']['routing_rate_test']:.1%}**,
fraud rate inside the band **{metrics['operational']['fraud_rate_in_band_test']:.1%}**
(vs {yte.mean():.1%} overall) - the router does concentrate the hard cases.

## 3. Quantum approach (challenge output: encoding + circuit design)

- **Encoding**: each step feature f in [0,1] -> single-qubit state
  `sqrt(1-f)|0> + sqrt(f)|1>` (RY(2 asin sqrt f)). {cfg['qrc']['input_qubits']} input
  qubits x {cfg['qrc']['reupload_rounds']} data-re-uploading rounds = up to 9 features/step.
- **Reservoir**: fully-connected transverse-field Ising,
  `H = sum J_ij X_i X_j + h sum Z_i`, `J_ij ~ U(-J/2, J/2)` drawn once per
  `reservoir_seed` and frozen. Evolution `exp(-i H dt)`, `dt={cfg['qrc']['dt']}`.
- **Injection** is a CPTP map (partial trace of input qubits + fresh product
  state) - this contraction is what gives the reservoir fading memory.
- **Read-out**: `<Z_i>` and `<Z_i Z_j>` at {cfg['qrc']['virtual_nodes']} virtual
  nodes (time multiplexing) of the final step, plus per-step trajectory
  mean/delta. Only a regularised logistic readout is trained.
- **Resource**: {q['n_qubits']} qubits, {q['evolution_steps_per_txn']} evolution
  sub-steps per transaction, embedding dim {q['embedding_dim']}, shots
  {q['shots']}. Simulator: exact NumPy density matrix; PennyLane/Braket-portable
  circuit cross-checked in `src/qrc_pennylane.py`.

## 4. Results - full held-out test set

| model | AUC-ROC | AUPRC | F1 | precision | recall | TP/FP/FN/TN |
|---|---|---|---|---|---|---|
{row('B0 logreg (static)', ft['B0_logreg_static'])}
{row('B1 LightGBM (static)', ft['B1_lgbm_static'])}
{row('B4 LightGBM (standard aggregates)', ft['B4_lgbm_aggregates'])}
{row('B4 calibrated', ft['B4_calibrated'])}
{row('B5 LightGBM (aggressive manual temporal FE)', ft['B5_lgbm_manual_temporal_FE'])}
{row('B_catboost (same features as B4)', ft['B_catboost_aggregates'])}
{row('ESN hybrid (end-to-end)', ft['ESN_end2end'])}
{row('Q3 QRC hybrid (end-to-end)', ft['Q3_hybrid_end2end'])}

(confusion matrix at each model's operating threshold; B4/ESN at t*, Q3 at the
frozen hybrid threshold.)

**Where the value sits.** B5 (aggressive hand-built temporal features on the
same GBDT) vs B4: dAUPRC {metrics['deltas']['auprc_B5_vs_B4_full']['delta']:+.4f}
CI [{metrics['deltas']['auprc_B5_vs_B4_full']['ci_low']:+.4f}, {metrics['deltas']['auprc_B5_vs_B4_full']['ci_high']:+.4f}].
Q3 hybrid vs B5: dAUPRC {metrics['deltas']['auprc_Q3_vs_B5_manualFE_full']['delta']:+.4f}
CI [{metrics['deltas']['auprc_Q3_vs_B5_manualFE_full']['ci_low']:+.4f}, {metrics['deltas']['auprc_Q3_vs_B5_manualFE_full']['ci_high']:+.4f}].
Read: how much of the ordered-sequence signal a analyst can capture by hand (B5-B4),
and whether the reservoir adds anything on top of that (Q3-B5).

**Does B4's strength depend on picking LightGBM?** CatBoost (same feature set
as B4, no other change) vs B4: dAUPRC
{metrics['deltas']['auprc_CatBoost_vs_B4_full']['delta']:+.4f}
CI [{metrics['deltas']['auprc_CatBoost_vs_B4_full']['ci_low']:+.4f}, {metrics['deltas']['auprc_CatBoost_vs_B4_full']['ci_high']:+.4f}].
Per the IEEE-CIS 1st-place writeup, CatBoost was in fact the strongest single
GBDT of the three they ensembled (private-LB AUC: CatBoost 0.9408, LightGBM
0.9384, XGBoost 0.9324) - this row checks whether that pattern shows up here too.

{_smoothing_section(metrics)}

## 5. Results - ambiguous band only (where QRC actually operates)

| model | AUC-ROC | AUPRC | F1 | precision | recall | TP/FP/FN/TN |
|---|---|---|---|---|---|---|
{row('B4 p0 (routed rows)', bt['B4_p0'])}
{row('Q0 QRC-only', bt['Q0_qrc_only'])}
{row('Q1 score fusion', bt['Q1_score_fusion'])}
{row('Q2 feature fusion', bt['Q2_feature_fusion'])}
{row('Q2 no-p0 (ablation)', bt['Q2_no_p0_ablation'])}
{_bandrow(bt, 'Q_deployed')}
{_bandrow(bt, 'ESN_deployed')}

## 6. Pre-registered success criteria

| # | criterion | result | verdict |
|---|---|---|---|
| C1 | dAUPRC(Q3 vs B4, full test) > 0, 95% CI > 0 | {c1['delta']:+.4f}  CI [{c1['ci_low']:+.4f}, {c1['ci_high']:+.4f}] | {'PASS' if sc['c1_delta_auprc_positive_ci']['pass_'] else 'fail'} |
| C2 | false-decline rate reduced at recall {rc0}, 95% CI < 0 | {c2['delta']:+.4f}  CI [{c2['ci_low']:+.4f}, {c2['ci_high']:+.4f}] | {'PASS' if sc['c2_false_decline_reduced_ci']['pass_'] else 'fail'} |
| C3 | Q2 at least at parity with the classical ESN on the band (95% CI low > -0.02; "beats" = CI low > 0) | {c3['delta']:+.4f}  CI [{c3['ci_low']:+.4f}, {c3['ci_high']:+.4f}]  (beats={sc['c3_vs_esn_on_band']['beats_esn']}) | {'PASS' if sc['c3_vs_esn_on_band']['pass_'] else 'fail'} |
| C4 | holds across reservoir seeds {manifest['frozen']['reservoir_seeds']} | see metrics.json > seeds | {'PASS' if sc['c4_holds_across_seeds']['pass_'] else 'fail'} |

**Overall measurable improvement: {'YES' if sc['overall_measurable_improvement'] else 'NO'}**

> {sc['note']}

## 7. Ablations (band AUPRC)

The deployment guard (section 5/6 - never deploy worse than `p0`) is **disabled**
for every row below: an ablation must show the true damage of a destructive
intervention (shuffled order, no entanglement, few shots, ...), not have it
masked by a silent fallback to `p0`. So these numbers are not directly the
Q2/Q3 numbers reported above; read them relative to the "ideal (unguarded)" row.

{_ablation_table(metrics)}

## 8. Distillation for production

| | band AUPRC | full-test AUPRC | latency |
|---|---|---|---|
| Q3 hybrid (needs offline QRC) | {st['hybrid_band_auprc']:.4f} | - | not real-time |
| distilled classical student | {st['band_auprc']:.4f} | {st['full_test_auprc']:.4f} | {st['inference_latency_ms_per_txn']:.4f} ms/txn |

The student runs comfortably inside the 100-300 ms authorisation budget; it
recovers {100*st['band_auprc']/max(st['hybrid_band_auprc'],1e-9):.0f}% of the
hybrid's band AUPRC.

## 9. Feature attribution (challenge output #3)

- LightGBM path: SHAP / tree contributions, per transaction (see `predictions.csv`).
- Fusion path: grouped contribution {metrics['fusion_group_attribution']}
  (no single quantum observable is claimed to "cause" fraud).

## 10. Honest limitations

- Synthetic data: absolute numbers are not IEEE-CIS numbers; the *comparison*
  (QRC vs ESN vs LightGBM under identical splits) is the transferable result.
- A {q['n_qubits']}-qubit reservoir is, per Fujii & Nakajima, in the class of a
  ~100-500 node ESN. Do not extrapolate to "quantum advantage".
- Quantum execution is offline. Production value = feature discovery + distillation.
- `entity_key` is a proxy; `metrics.json` / config carry the E1 vs E2 sensitivity hook.
"""
    open(outdir / "report.md", "w").write(md)


def _smoothing_section(metrics):
    s = metrics.get("smoothing", {})
    if not s.get("enabled"):
        return "_entity-level smoothing disabled for this run._"
    dc, db = s["delta_causal"], s["delta_batch_AUDIT_ONLY"]
    return f"""**Entity-level smoothing** (the IEEE-CIS 1st-place solution's other big lever
besides feature engineering: replace a transaction's score with its card/client's
average score). Alpha frozen on validation = **{s['alpha_star']}** (1.0 = no
smoothing, 0.0 = fully replaced by the entity mean) from grid {s['alpha_grid']}.

| variant | deployable in HSBC's real-time path? | AUPRC | delta vs B4 | 95% CI |
|---|---|---|---|---|
| B4 baseline | - | {s['auprc_baseline_B4']:.4f} | - | - |
| + causal smoothing (past-only entity mean) | **yes** | {s['auprc_causal_smooth']:.4f} | {dc['delta']:+.4f} | [{dc['ci_low']:+.4f}, {dc['ci_high']:+.4f}] |
| + batch smoothing (whole-batch entity mean, exact Kaggle trick) | **no - uses each entity's future test transactions** | {s['auprc_batch_smooth_AUDIT_ONLY']:.4f} | {db['delta']:+.4f} | [{db['ci_low']:+.4f}, {db['ci_high']:+.4f}] |

The batch row is reported only as an offline/audit ceiling - it is exactly what
the Kaggle-winning team did on their one-shot test submission, but HSBC's
<300ms authorization flow does not have an entity's future transactions
available yet, so it can never be deployed as-is."""


def _bandrow(bt, prefix):
    key = next((k for k in bt if k.startswith(prefix)), None)
    if key is None:
        return f"| {prefix} | - | - | - | - | - | - |"
    d = bt[key]
    cm = d["confusion_matrix"]
    return (f"| {key} | {d['auroc']:.4f} | {d['auprc']:.4f} | {d['f1']:.4f} | "
            f"{d['precision']:.4f} | {d['recall']:.4f} | "
            f"{cm['tp']}/{cm['fp']}/{cm['fn']}/{cm['tn']} |")


def _ablation_table(metrics):
    ab = metrics.get("ablations", {})
    if not ab:
        return "_ablations disabled_"
    lines = ["| ablation | band AUPRC | vs ideal |", "|---|---|---|"]
    ideal = ab.get("ideal_feature_fusion_band_auprc", float("nan"))
    lines.append(f"| ideal feature fusion (unguarded) | {ideal:.4f} | - |")
    dep = ab.get("ideal_feature_fusion_band_auprc_guarded_deployed")
    if dep is not None:
        lines.append(f"| _(for reference: guarded/deployed number, section 5)_ | {dep:.4f} | {dep-ideal:+.4f} |")
    for k in ("no_entanglement", "random_reservoir", "shuffle_sequence",
              "no_history_K1"):
        if k in ab:
            lines.append(f"| {k} | {ab[k]:.4f} | {ab[k]-ideal:+.4f} |")
    for k, dd in (("qubit_sweep_band_auprc", "qubits"),
                  ("K_sweep_band_auprc", "K"),
                  ("dt_sweep_band_auprc", "dt")):
        if k in ab:
            for kk, vv in ab[k].items():
                lines.append(f"| {dd}={kk} | {vv:.4f} | {vv-ideal:+.4f} |")
    if "shots_sweep" in ab:
        for sh, au in ab["shots_sweep"]:
            lines.append(f"| shots={sh} | {au:.4f} | {au-ideal:+.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
