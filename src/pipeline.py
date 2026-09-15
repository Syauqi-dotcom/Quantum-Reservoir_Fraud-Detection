"""
End-to-end PoC pipeline for *Abstraction*
(Hybrid LightGBM + Quantum Reservoir Computing for card-not-present fraud).

The pipeline follows the staged architecture of ``../Abstraction.md``.  Each
stage lives in its own ``src/`` module; this file only wires them together and
writes the challenge deliverables under ``results/``:

    Stage A   data + leakage-safe features + chronological split
              -> data.py / data_ieee.py, features.py
    Stage A*  classical baselines B0/B1/B4/B5/CatBoost + isotonic calibration -> p0
              -> baselines.py, calibration.py
    Stage H   entity-level score smoothing (independent classical lever)
              -> postprocess.py
    Stage B   ambiguity router  (p0 in [tau_low, tau_high] -> QRC second opinion)
              -> routing.py
    Stage C   per-entity sequence builder for the routed band
              -> sequence.py
    Stage D   fixed transverse-field-Ising quantum reservoir (+ classical ESN control)
              -> qrc.py, esn.py
    Stage E   logistic fusion / readout + rank-preserving splice back into p0
              -> fusion.py
    Stage F   distillation to a real-time classical student
              -> distill.py
    eval      metrics, paired-bootstrap success criteria, ablations, attribution, plots
              -> evaluate.py, attribution.py, plots.py, report.py

Nothing here trains the quantum circuit; only the downstream logistic readout
is fitted (Abstraction.md: the quantum circuit keeps "parameter yang sebagian
besar tetap").

    from src.pipeline import Experiment
    Experiment(cfg, outdir).run()
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from . import attribution as attr
from . import baselines as base
from . import data as datamod
from . import esn as esnmod
from . import evaluate as ev
from . import features as feat
from . import plots as plt_
from . import postprocess as post
from . import report
from . import routing as route
from . import sequence as seqmod
from .calibration import Calibrator, threshold_for_recall
from .distill import train_student
from .fusion import FusionModel, qrc_only_score
from .qrc import QuantumReservoir, make_embeddings

_PACKAGES = ("numpy", "pandas", "sklearn", "lightgbm", "catboost",
             "scipy", "shap", "pennylane")


# ===========================================================================
# Stage A -- feature assembly helpers
# ===========================================================================
def build_features(train, val, test, entity_cols, extra_cols=()):
    """Attach the entity key, build static + past-only rolling features, and
    return the three GBDT feature views (B4 = standard aggregates, B5 = B4 +
    aggressive manual temporal FE, ``none`` = the deliberately-minimal static
    block used by B0/B1 and the sequence path)."""
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
        cols = {k: feat.assemble(fmap[k], rolling, ids[k], rolling=kind)
                for k in ("train", "val", "test")}
        # the raw IEEE feature block goes to the GBDT baselines (b4 / b5), not to
        # the deliberately-minimal static logreg/GBDT (b1) or the sequence path.
        return _augment(cols) if kind in ("b4", "b5") else cols

    return (train, val, test, full, rolling, X("b4"), X("b5"), X("none"))


class EmbReducer:
    """Standardise + PCA an (n, D) reservoir embedding, fit on train rows only.
    Keeps the logistic read-out well-conditioned when D >> #positives."""

    def __init__(self, k=28):
        self.k = k
        self._sc = None
        self._pca = None

    def fit(self, Z, tr_idx):
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


def fit_fusion_variant(variant, C_grid, p0, z, ctx, y, split_idx,
                       z_score=None, guard=True):
    """Fit one Stage-E fusion head and return (model, val prediction, test prediction)."""
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


def _rank_splice(p0_split, band_mask, band_scores):
    """Rank-preserving splice: the band keeps its exact p0 score interval, but
    rows inside it are re-ordered according to the fusion head -- so full-test
    AUPRC can only move via a genuinely better within-band ranking."""
    p = np.asarray(p0_split, float).copy()
    target = np.sort(p[band_mask])
    order = np.argsort(np.argsort(band_scores))       # ranks 0..m-1
    p[band_mask] = target[order]
    return p


# ===========================================================================
# the pipeline
# ===========================================================================
class Experiment:
    def __init__(self, cfg: dict, outdir, t0: float | None = None):
        self.cfg = cfg
        self.seed = cfg["run"]["seed"]
        self.t0 = t0 if t0 is not None else time.time()
        self.outdir = Path(outdir)
        (self.outdir / "plots").mkdir(parents=True, exist_ok=True)
        cfg["qrc"]["_K_traj"] = cfg["sequence"]["K"]   # trajectory length in the embedding

        self.manifest = dict(
            config=cfg,
            python=sys.version.split()[0],
            platform=platform.platform(),
            packages={},
            started=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        for p in _PACKAGES:
            try:
                self.manifest["packages"][p] = __import__(p).__version__
            except Exception:
                self.manifest["packages"][p] = None

    def log(self, msg):
        print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)

    # -- driver ------------------------------------------------------------
    def run(self):
        self.load_data()
        self.split_and_features()
        self.train_baselines()
        self.entity_smoothing()
        self.fit_router()
        self.build_band_sequences()
        self.embed_reservoirs()
        self.fit_fusion()
        self.reconstruct_end_to_end()
        self.compute_metrics()
        self.multi_seed_check()
        self.run_ablations()
        self.distill_student()
        self.write_predictions()
        self.evaluate_success_criteria()
        self.finalize()
        return self.metrics, self.sc

    # ===================================================================
    # Stage A -- data, features, chronological split
    # ===================================================================
    def load_data(self):
        cfg, seed = self.cfg, self.seed
        if cfg["run"].get("data_source") == "ieee":
            from . import data_ieee as ieeemod
            self.log("loading IEEE-CIS Fraud Detection (real Kaggle csv) ...")
            df = ieeemod.load(cfg, seed)
        else:
            self.log("generating synthetic CNP transactions ...")
            df = datamod.generate(cfg, seed)
        self.extra_cols = list(cfg["data"].get("_extra_cols", []))
        self.df = df
        self.log(f"  {len(df):,} transactions | fraud rate {df.isFraud.mean():.3%} | "
                 f"{df.entity.nunique():,} entities | {len(self.extra_cols)} raw IEEE features")
        self.log(f"  fraud composition: "
                 f"{df[df.isFraud == 1].fraud_type.value_counts().to_dict()}")

    def split_and_features(self):
        cfg, df = self.cfg, self.df
        self.primary_key = cfg["data"]["entity_keys"]["E2"]
        train, val, test = datamod.chronological_split(df, cfg)
        self.log(f"  split: train {len(train):,} | val {len(val):,} | test {len(test):,}")

        (self.train, self.val, self.test, self.full, self.rolling,
         self.Xb4, self.Xb5, self.Xst) = build_features(
            train, val, test, self.primary_key, self.extra_cols)
        self.Xtr, self.Xva, self.Xte = self.Xb4["train"], self.Xb4["val"], self.Xb4["test"]

        self.ytr = self.train.isFraud.to_numpy()
        self.yva = self.val.isFraud.to_numpy()
        self.yte = self.test.isFraud.to_numpy()

    # ===================================================================
    # Stage A* -- classical baselines + calibration -> p0
    # ===================================================================
    def train_baselines(self):
        cfg, seed = self.cfg, self.seed
        Xst, Xb5 = self.Xst, self.Xb5
        Xtr, Xva, Xte = self.Xtr, self.Xva, self.Xte
        ytr, yva = self.ytr, self.yva
        self.log("training baselines ...")

        b0 = base.train_logreg(Xst["train"].to_numpy(), ytr)
        b1 = base.train_gbdt(Xst["train"], ytr, cfg["baselines"]["lightgbm"], seed)
        b4 = base.train_gbdt(Xtr, ytr, cfg["baselines"]["lightgbm"], seed)
        b5 = base.train_gbdt(Xb5["train"], ytr, cfg["baselines"]["lightgbm"], seed)
        # CatBoost on the SAME feature set as B4 - per the IEEE-CIS 1st-place
        # writeup it was the strongest single GBDT of the three they ensembled
        # (0.9408 vs LightGBM 0.9384, XGBoost 0.9324 private-LB AUC), so this is
        # the fair "does B4's strength depend on picking LightGBM?" sanity check.
        b_cat = base.train_catboost(Xtr, ytr, cfg["baselines"].get("catboost", {}), seed)
        self.b4 = b4

        self.p_b0 = dict(val=b0.predict_proba(Xst["val"].to_numpy())[:, 1],
                         test=b0.predict_proba(Xst["test"].to_numpy())[:, 1])
        self.p_b1 = dict(val=base.predict_proba(b1, Xst["val"]),
                         test=base.predict_proba(b1, Xst["test"]))
        self.p_b5 = dict(val=base.predict_proba(b5, Xb5["val"]),
                         test=base.predict_proba(b5, Xb5["test"]))
        self.p_cat = dict(val=base.predict_proba(b_cat, Xva),
                          test=base.predict_proba(b_cat, Xte))
        # out-of-fold train scores so the train score distribution matches val/test
        # (an in-sample LightGBM score is wildly overconfident and breaks routing).
        oof = cross_val_predict(
            base.make_gbdt(cfg["baselines"]["lightgbm"], seed), Xtr.to_numpy(), ytr,
            cv=StratifiedKFold(4, shuffle=True, random_state=seed),
            method="predict_proba", n_jobs=1)[:, 1]
        self.p_b4_raw = dict(train=oof,
                             val=base.predict_proba(b4, Xva),
                             test=base.predict_proba(b4, Xte))

        self.cal = Calibrator(cfg["baselines"]["calibration"]).fit(self.p_b4_raw["val"], yva)
        self.p0 = dict(train=self.cal.transform(self.p_b4_raw["train"]),
                       val=self.cal.transform(self.p_b4_raw["val"]),
                       test=self.cal.transform(self.p_b4_raw["test"]))

        # calibrated p0 for every row (used as the lagged hist_score in sequences)
        self.p0_all = {}
        for split_df, key in ((self.train, "train"), (self.val, "val"), (self.test, "test")):
            for tid, pv in zip(split_df["TransactionID"].to_numpy(), self.p0[key]):
                self.p0_all[int(tid)] = float(pv)

    # ===================================================================
    # Stage H -- entity-level smoothing (independent classical lever, kept
    # OUT of the routing/QRC path so it stays a clean one-factor experiment)
    # ===================================================================
    def entity_smoothing(self):
        cfg, seed = self.cfg, self.seed
        train, val, test, full = self.train, self.val, self.test, self.full
        yte = self.yte
        self.smoothing_result = {"enabled": False}
        if not cfg.get("smoothing", {}).get("enabled", True):
            return

        self.log("entity-level smoothing (IEEE-CIS 1st-place style) ...")
        split_of = {}
        for tid in train["TransactionID"]: split_of[int(tid)] = "train"
        for tid in val["TransactionID"]: split_of[int(tid)] = "val"
        for tid in test["TransactionID"]: split_of[int(tid)] = "test"
        full_p0 = full[["TransactionID", "entity_key", "isFraud"]].copy()
        full_p0["p0"] = full_p0["TransactionID"].map(self.p0_all)
        full_p0["split"] = full_p0["TransactionID"].map(split_of)
        val_mask = (full_p0["split"] == "val").to_numpy()
        test_mask = (full_p0["split"] == "test").to_numpy()

        alpha_grid = cfg["smoothing"].get("alpha_grid", [1.0, 0.75, 0.5, 0.25, 0.0])
        alpha_star = post.select_alpha(full_p0, "p0", "entity_key", "isFraud",
                                       val_mask, alpha_grid)
        causal_full = post.causal_smooth(full_p0, "p0", "entity_key", alpha_star)
        p0_causal_test = causal_full[test_mask]
        p0_batch_test = post.batch_smooth(full_p0[test_mask], "p0", "entity_key")

        bi = cfg["eval"]["bootstrap_iters"]
        ap_base = float(ev.auprc_fn(yte, self.p0["test"]))
        ap_causal = float(ev.auprc_fn(yte, p0_causal_test))
        ap_batch = float(ev.auprc_fn(yte, p0_batch_test))
        d_causal = ev.paired_bootstrap_delta(yte, p0_causal_test, self.p0["test"],
                                             ev.auprc_fn, iters=bi, seed=seed)
        d_batch = ev.paired_bootstrap_delta(yte, p0_batch_test, self.p0["test"],
                                            ev.auprc_fn, iters=bi, seed=seed)
        self.smoothing_result = dict(
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
        self.log(f"  alpha* (frozen on validation) = {alpha_star}")
        self.log(f"  B4 baseline AUPRC            = {ap_base:.4f}")
        self.log(f"  + causal smoothing (real-time)= {ap_causal:.4f}  "
                 f"delta={d_causal['delta']:+.4f} "
                 f"CI=[{d_causal['ci_low']:+.4f},{d_causal['ci_high']:+.4f}]")
        self.log(f"  + batch smoothing (AUDIT ONLY)= {ap_batch:.4f}  "
                 f"delta={d_batch['delta']:+.4f} "
                 f"CI=[{d_batch['ci_low']:+.4f},{d_batch['ci_high']:+.4f}]")

    # ===================================================================
    # Stage B -- ambiguity router (FROZEN on validation, applied once to test)
    # ===================================================================
    def fit_router(self):
        cfg = self.cfg
        yva, ytr = self.yva, self.ytr
        self.log("fitting ambiguity band on validation ...")
        band = route.fit_band(yva, self.p0["val"],
                              cfg["routing"]["target_fraud_recall"],
                              cfg["routing"]["review_recall"],
                              cfg["routing"]["routing_budget"],
                              y_train=ytr, p0_train=self.p0["train"])
        cfg["routing"]["tau_low"] = band["tau_low"]
        cfg["routing"]["tau_high"] = band["tau_high"]
        self.band = band
        self.log(f"  band = [{band['tau_low']:.4f}, {band['tau_high']:.4f}]  "
                 f"t*={band['t_star']:.4f}  routed(val)={band['routed_fraction_val']:.3%}  "
                 f"pos(train+val)={band['pos_in_band_trainval']}")

        self.amb = dict(train=route.apply_band(self.p0["train"], band),
                        val=route.apply_band(self.p0["val"], band),
                        test=route.apply_band(self.p0["test"], band))
        ymap = {"train": ytr, "val": yva, "test": self.yte}
        for k in self.amb:
            yy = ymap[k]
            self.log(f"  {k}: routed {self.amb[k].mean():.3%}  "
                     f"fraud-in-band {yy[self.amb[k]].mean():.3%}  "
                     f"(vs {yy.mean():.3%} overall)")

    # ===================================================================
    # Stage C -- per-entity sequences for the ambiguous band
    # ===================================================================
    def build_band_sequences(self):
        cfg = self.cfg
        ytr, yva, yte = self.ytr, self.yva, self.yte
        self.log("building per-entity sequences for the ambiguous band ...")
        self.K = cfg["sequence"]["K"]
        split_df = {"train": self.train, "val": self.val, "test": self.test}
        self.ids = {k: split_df[k]["TransactionID"].to_numpy()[self.amb[k]] for k in self.amb}

        seqs, masks, lens = {}, {}, {}
        for k in ("train", "val", "test"):
            s, m, L = seqmod.build_sequences(self.full, self.rolling, self.p0_all,
                                             self.ids[k], self.K)
            seqs[k], masks[k], lens[k] = s, m, L
        sc = seqmod.SeqScaler().fit(seqs["train"], masks["train"])
        seqs_s = {k: sc.transform(seqs[k]) for k in seqs}
        ymap = {"train": ytr, "val": yva, "test": yte}
        self.y_amb = {k: ymap[k][self.amb[k]] for k in self.amb}
        for k in ("train", "val", "test"):
            self.log(f"  band {k}: n={self.amb[k].sum()}  fraud={int(self.y_amb[k].sum())}")
        if self.y_amb["train"].sum() < 3 or self.y_amb["val"].sum() < 3:
            raise SystemExit("Ambiguous band has too few fraud examples to train the "
                             "fusion read-out. Increase routing_budget / data size.")

        self.ctx_names = cfg["fusion"]["context_features"]
        self.ctx = {k: ctx_matrix(self.rolling, lens[k], self.ids[k], self.ctx_names)
                    for k in self.ids}

        n_tr, n_va, n_te = (len(self.ids["train"]), len(self.ids["val"]),
                            len(self.ids["test"]))
        self.split_idx = (np.arange(n_tr),
                          np.arange(n_tr, n_tr + n_va),
                          np.arange(n_tr + n_va, n_tr + n_va + n_te))

        def stack(d):
            return np.concatenate([d["train"], d["val"], d["test"]], axis=0)

        self.seq_all = stack(seqs_s)
        self.mask_all = np.concatenate([masks["train"], masks["val"], masks["test"]], axis=0)
        self.y_all = np.concatenate([self.y_amb["train"], self.y_amb["val"], self.y_amb["test"]])
        self.p0_all_band = np.concatenate(
            [self.p0["train"][self.amb["train"]], self.p0["val"][self.amb["val"]],
             self.p0["test"][self.amb["test"]]])
        self.ctx_all = np.vstack([self.ctx["train"], self.ctx["val"], self.ctx["test"]])

    # ===================================================================
    # Stage D -- quantum reservoir (main run) + classical ESN control
    # ===================================================================
    def embed_reservoirs(self):
        cfg = self.cfg
        self.seeds = cfg["qrc"]["reservoir_seeds"]

        t_qrc = time.time()
        qcfg = {**cfg["qrc"]}
        self.log(f"  QRC embed [main] n={len(self.seq_all)} qubits={qcfg['n_qubits']} "
                 f"M={qcfg.get('spatial_multiplex', 1)} shots={qcfg.get('shots')}")
        Z_qrc = make_embeddings(qcfg, self.seeds, self.seq_all, self.mask_all)
        self.qrc_secs = time.time() - t_qrc
        self.log(f"  QRC embed done in {self.qrc_secs:.1f}s  dim={Z_qrc.shape[1]}")

        esn = esnmod.EchoStateNetwork(cfg["esn"], self.seq_all.shape[-1], self.seed)
        Z_esn = esn.embed(self.seq_all, self.mask_all)
        self.log(f"  ESN embed done  dim={Z_esn.shape[1]}")

        # PCA-reduce both reservoir embeddings (fit on train band only) so the
        # logistic read-out stays well-conditioned given ~10^2-10^3 positives.
        self.tr, self.va, self.te = self.split_idx
        self.kdim = int(cfg["fusion"].get("embed_pca_dim", 28))
        self.red_qrc = EmbReducer(self.kdim).fit(Z_qrc, self.tr)
        self.red_esn = EmbReducer(self.kdim).fit(Z_esn, self.tr)
        self.Z_qrc = self.red_qrc.transform(Z_qrc)
        self.Z_esn = self.red_esn.transform(Z_esn)
        self.log(f"  reduced embeddings -> QRC {self.Z_qrc.shape[1]}  ESN {self.Z_esn.shape[1]}")

    # ===================================================================
    # Stage E -- fusion / readout variants (QRC + ESN control)
    # ===================================================================
    def fit_fusion(self):
        cfg = self.cfg
        tr, va, te = self.tr, self.va, self.te
        y_all = self.y_all
        self.Cg = cfg["fusion"]["C_grid"]
        self.log("fitting fusion / readout variants ...")

        zs_tr, zs_va, zs_te = qrc_only_score(
            self.Z_qrc[tr], y_all[tr], self.Z_qrc[va], y_all[va], self.Z_qrc[te], self.Cg)
        self.z_score_all = np.concatenate([zs_tr, zs_va, zs_te])

        self.fusion_preds = {}
        self.fusion_models = {}
        for variant in cfg["fusion"]["variants"]:
            fm, pv, pt = fit_fusion_variant(
                variant, self.Cg, self.p0_all_band, self.Z_qrc, self.ctx_all,
                y_all, self.split_idx, z_score=self.z_score_all)
            self.fusion_models[variant] = fm
            self.fusion_preds[variant] = dict(val=pv, test=pt, C=fm.C_)
            self.log(f"  {variant:22s}  C={fm.C_}  test-band AUPRC="
                     f"{ev.auprc_fn(y_all[te], pt):.4f}")

        # ESN control: fit ALL the same fusion heads, deploy the same one QRC deploys
        self.esn_fits = {}
        for variant in ("qrc_only", "score_fusion", "feature_fusion"):
            zs = qrc_only_score(self.Z_esn[tr], y_all[tr], self.Z_esn[va], y_all[va],
                                self.Z_esn[te], self.Cg) \
                if variant == "score_fusion" else (None, None, None)
            zsc = np.concatenate(zs) if zs[0] is not None else None
            fmE, pvE, ptE = fit_fusion_variant(
                variant, self.Cg, self.p0_all_band, self.Z_esn, self.ctx_all,
                y_all, self.split_idx, z_score=zsc)
            self.esn_fits[variant] = dict(model=fmE, val=pvE, test=ptE)
            self.log(f"  ESN {variant:18s} C={fmE.C_}  test-band AUPRC="
                     f"{ev.auprc_fn(y_all[te], ptE):.4f}")

    # ===================================================================
    # end-to-end reconstruction on test + per-model decision thresholds
    # ===================================================================
    def reconstruct(self, band_pred_test):
        return _rank_splice(self.p0["test"], self.amb["test"], band_pred_test)

    def reconstruct_end_to_end(self):
        cfg = self.cfg
        va = self.va
        y_all, yva = self.y_all, self.yva
        # pick the deployed fusion head on VALIDATION-band AUPRC (model selection,
        # never on test) among the p0-aware variants.  score_fusion (Q1) is the
        # architecture in Abstraction.md and the robust default; only deploy the
        # richer feature_fusion if it is clearly better on the validation band.
        ap_sf = ev.auprc_fn(y_all[va], self.fusion_preds["score_fusion"]["val"])
        ap_ff = ev.auprc_fn(y_all[va], self.fusion_preds["feature_fusion"]["val"])
        self.q3_variant = "feature_fusion" if ap_ff > ap_sf + 0.015 else "score_fusion"
        self.log(f"  deployed fusion head = {self.q3_variant}  "
                 f"(val AUPRC score_fusion={ap_sf:.4f} feature_fusion={ap_ff:.4f})")
        esn_dep = self.esn_fits[self.q3_variant]
        self.fm_esn, self.pv_esn, self.pt_esn = (esn_dep["model"], esn_dep["val"],
                                                 esn_dep["test"])

        self.p_q3_e2e = self.reconstruct(self.fusion_preds[self.q3_variant]["test"])
        self.p_esn_e2e = self.reconstruct(self.pt_esn)

        # per-model decision thresholds: frozen on validation end-to-end scores
        self.rc_t = cfg["routing"]["target_fraud_recall"]
        p_q3_val = _rank_splice(self.p0["val"], self.amb["val"],
                                self.fusion_preds[self.q3_variant]["val"])
        p_esn_val = _rank_splice(self.p0["val"], self.amb["val"], self.pv_esn)
        self.t_hybrid = threshold_for_recall(yva, p_q3_val, self.rc_t)
        self.t_esn = threshold_for_recall(yva, p_esn_val, self.rc_t)
        self.log(f"  frozen hybrid decision threshold (val, recall {self.rc_t}) "
                 f"= {self.t_hybrid:.4f}")

    # ===================================================================
    # metrics
    # ===================================================================
    def compute_metrics(self):
        cfg, seed = self.cfg, self.seed
        yte, yva = self.yte, self.yva
        y_all = self.y_all
        tr, va, te = self.tr, self.va, self.te
        fp = self.fusion_preds
        self.log("computing metrics ...")

        t_b4 = threshold_for_recall(yva, self.p0["val"], self.rc_t)
        self.t_b4 = t_b4
        rc_t = self.rc_t
        metrics = {"full_test": {}, "ambiguous_band_test": {}, "deltas": {},
                   "seeds": {}, "ablations": {}, "calibration": {}, "operational": {},
                   "smoothing": self.smoothing_result}

        t_b5 = threshold_for_recall(yva, self.p_b5["val"], rc_t)
        t_cat = threshold_for_recall(yva, self.p_cat["val"], rc_t)
        full_models = {
            "B0_logreg_static": self.p_b0["test"],
            "B1_lgbm_static": self.p_b1["test"],
            "B4_lgbm_aggregates": self.p_b4_raw["test"],
            "B4_calibrated": self.p0["test"],
            "B5_lgbm_manual_temporal_FE": self.p_b5["test"],
            "B_catboost_aggregates": self.p_cat["test"],
            "ESN_end2end": self.p_esn_e2e,
            "Q3_hybrid_end2end": self.p_q3_e2e,
        }
        thr_map = {"B0_logreg_static": 0.5, "B1_lgbm_static": 0.5,
                   "B4_lgbm_aggregates": t_b4, "B4_calibrated": t_b4,
                   "B5_lgbm_manual_temporal_FE": t_b5, "B_catboost_aggregates": t_cat,
                   "ESN_end2end": self.t_esn, "Q3_hybrid_end2end": self.t_hybrid}
        for nm, p in full_models.items():
            metrics["full_test"][nm] = ev.basic_metrics(yte, p, thr_map.get(nm, 0.5))

        band_models = {
            "B4_p0": (self.p0["test"][self.amb["test"]], self.p0["val"][self.amb["val"]]),
            "Q0_qrc_only": (fp["qrc_only"]["test"], fp["qrc_only"]["val"]),
            "Q1_score_fusion": (fp["score_fusion"]["test"], fp["score_fusion"]["val"]),
            "Q2_feature_fusion": (fp["feature_fusion"]["test"], fp["feature_fusion"]["val"]),
            "Q2_no_p0_ablation": (fp["feature_fusion_no_p0"]["test"],
                                  fp["feature_fusion_no_p0"]["val"]),
            f"Q_deployed[{self.q3_variant}]": (fp[self.q3_variant]["test"],
                                               fp[self.q3_variant]["val"]),
            f"ESN_deployed[{self.q3_variant}]": (self.pt_esn, self.pv_esn),
        }
        for nm, (p_te, p_va) in band_models.items():
            thr = threshold_for_recall(y_all[va], p_va, rc_t)
            metrics["ambiguous_band_test"][nm] = ev.basic_metrics(y_all[te], p_te, thr)
        band_models = {k: v[0] for k, v in band_models.items()}
        self._band_models = band_models

        # paired bootstrap deltas
        bi = cfg["eval"]["bootstrap_iters"]
        metrics["deltas"]["c1_auprc_Q3_vs_B4_full"] = ev.paired_bootstrap_delta(
            yte, self.p_q3_e2e, self.p_b4_raw["test"], ev.auprc_fn, iters=bi, seed=seed)
        for rc in cfg["eval"]["fixed_recall_points"]:
            fdr_fn = ev.fdr_fn_factory(rc)
            d = ev.paired_bootstrap_delta(yte, self.p_q3_e2e, self.p_b4_raw["test"],
                                          fdr_fn, iters=bi, seed=seed)
            metrics["deltas"][f"c2_fdr_Q3_vs_B4_recall{rc}"] = d
            fdr_q3, _ = ev.false_decline_rate(yte, self.p_q3_e2e, rc)
            fdr_b4, _ = ev.false_decline_rate(yte, self.p_b4_raw["test"], rc)
            metrics["operational"][f"false_decline_rate_recall{rc}"] = dict(
                B4=fdr_b4, Q3=fdr_q3, delta=fdr_q3 - fdr_b4)
        metrics["deltas"]["c3_auprc_Q2_vs_ESN_band"] = ev.paired_bootstrap_delta(
            y_all[te], fp[self.q3_variant]["test"], self.pt_esn, ev.auprc_fn,
            iters=bi, seed=seed)
        # honesty check: hybrid vs the aggressive manual-temporal-FE GBDT
        metrics["deltas"]["auprc_Q3_vs_B5_manualFE_full"] = ev.paired_bootstrap_delta(
            yte, self.p_q3_e2e, self.p_b5["test"], ev.auprc_fn, iters=bi, seed=seed)
        metrics["deltas"]["auprc_B5_vs_B4_full"] = ev.paired_bootstrap_delta(
            yte, self.p_b5["test"], self.p_b4_raw["test"], ev.auprc_fn, iters=bi, seed=seed)
        # sanity check: does B4's strength depend on picking LightGBM specifically?
        metrics["deltas"]["auprc_CatBoost_vs_B4_full"] = ev.paired_bootstrap_delta(
            yte, self.p_cat["test"], self.p_b4_raw["test"], ev.auprc_fn, iters=bi, seed=seed)

        metrics["calibration"]["B4_calibrated"] = dict(
            brier=metrics["full_test"]["B4_calibrated"]["brier"],
            ece=metrics["full_test"]["B4_calibrated"]["ece"])
        metrics["calibration"]["Q3_hybrid_end2end"] = dict(
            brier=metrics["full_test"]["Q3_hybrid_end2end"]["brier"],
            ece=metrics["full_test"]["Q3_hybrid_end2end"]["ece"])

        metrics["operational"]["routing_rate_test"] = float(self.amb["test"].mean())
        metrics["operational"]["fraud_rate_in_band_test"] = float(yte[self.amb["test"]].mean())
        self.metrics = metrics

    # ===================================================================
    # multi-seed check (C4)
    # ===================================================================
    def multi_seed_check(self):
        cfg, seed = self.cfg, self.seed
        yte, y_all = self.yte, self.y_all
        tr, va, te = self.tr, self.va, self.te
        bi = cfg["eval"]["bootstrap_iters"]
        self.log("multi-seed check ...")
        for s in self.seeds:
            Zs = make_embeddings({**cfg["qrc"], "spatial_multiplex": 1}, [s],
                                 self.seq_all, self.mask_all)
            Zs = EmbReducer(self.kdim).fit_transform(Zs, tr)
            zsc_s = None
            if self.q3_variant == "score_fusion":
                zsc_s = np.concatenate(qrc_only_score(Zs[tr], y_all[tr], Zs[va], y_all[va],
                                                      Zs[te], self.Cg))
            _, _, pt_s = fit_fusion_variant(self.q3_variant, self.Cg, self.p0_all_band,
                                            Zs, self.ctx_all, y_all, self.split_idx,
                                            z_score=zsc_s)
            pe = self.reconstruct(pt_s)
            d1 = ev.paired_bootstrap_delta(yte, pe, self.p_b4_raw["test"], ev.auprc_fn,
                                           iters=max(400, bi // 2), seed=s)
            fdr_fn = ev.fdr_fn_factory(cfg["routing"]["target_fraud_recall"])
            d2 = ev.paired_bootstrap_delta(yte, pe, self.p_b4_raw["test"], fdr_fn,
                                           iters=max(400, bi // 2), seed=s)
            db = ev.paired_bootstrap_delta(y_all[te], pt_s, self.pt_esn, ev.auprc_fn,
                                           iters=max(400, bi // 2), seed=s)
            self.metrics["seeds"][str(s)] = dict(
                delta_auprc_full=d1, delta_fdr_full=d2, delta_auprc_vs_esn_band=db,
                band_auprc=float(ev.auprc_fn(y_all[te], pt_s)))
            self.log(f"  seed {s}: dAUPRC(full)={d1['delta']:+.4f} [{d1['ci_low']:+.4f},"
                     f"{d1['ci_high']:+.4f}]  dFDR={d2['delta']:+.4f}  "
                     f"vs ESN band={db['delta']:+.4f}")

    # ===================================================================
    # ablations (band AUPRC; deployment guard disabled -- an ablation must
    # show the TRUE damage of a destructive intervention, never a masked
    # fallback to p0)
    # ===================================================================
    def _band_auprc_for(self, qcfg, seq_in=None, mask_in=None):
        seq_in = self.seq_all if seq_in is None else seq_in
        mask_in = self.mask_all if mask_in is None else mask_in
        Z = make_embeddings(qcfg, self.seeds[:1], seq_in, mask_in)
        Z = EmbReducer(self.kdim).fit_transform(Z, self.tr)
        _, _, pt = fit_fusion_variant("feature_fusion", self.Cg, self.p0_all_band, Z,
                                      self.ctx_all, self.y_all, self.split_idx, guard=False)
        return float(ev.auprc_fn(self.y_all[self.te], pt)), pt

    def run_ablations(self):
        cfg, seed = self.cfg, self.seed
        if not cfg["ablations"]["enabled"]:
            return
        self.log("ablations ...")
        A = cfg["ablations"]
        y_all, te = self.y_all, self.te
        mask_all, seq_all = self.mask_all, self.seq_all
        ab = self.metrics["ablations"]

        # unguarded ideal reference, reusing the already-computed main embedding
        # (no re-simulation: same reservoir_seeds[0], same PCA space)
        fm_ideal_ug, _, pt_ideal_ug = fit_fusion_variant(
            "feature_fusion", self.Cg, self.p0_all_band, self.Z_qrc, self.ctx_all,
            y_all, self.split_idx, guard=False)
        ideal_band_auprc = float(ev.auprc_fn(y_all[te], pt_ideal_ug))
        ab["ideal_feature_fusion_band_auprc"] = ideal_band_auprc
        ab["ideal_feature_fusion_band_auprc_guarded_deployed"] = float(
            ev.auprc_fn(y_all[te], self.fusion_preds["feature_fusion"]["test"]))

        if A["no_entanglement"]:
            ab["no_entanglement"], _ = self._band_auprc_for({**cfg["qrc"], "_entangle": False})
        if A["random_reservoir"]:
            ab["random_reservoir"], _ = self._band_auprc_for(
                {**cfg["qrc"], "_random_reservoir": True})
        if A["shuffle_sequence"]:
            rs = np.random.default_rng(seed)
            seq_sh = seq_all.copy()
            for i in range(len(seq_sh)):
                idx = np.where(mask_all[i] > 0)[0]
                perm = rs.permutation(idx)
                seq_sh[i, idx] = seq_sh[i, perm]
            ab["shuffle_sequence"], _ = self._band_auprc_for({**cfg["qrc"]}, seq_sh, mask_all)
        if A["no_history"]:
            m1 = mask_all.copy()
            for i in range(len(m1)):
                idx = np.where(m1[i] > 0)[0]
                if len(idx) > 1:
                    m1[i, idx[:-1]] = 0.0
            ab["no_history_K1"], _ = self._band_auprc_for({**cfg["qrc"]}, seq_all, m1)

        # shot-noise sweep: fusion trained on the ideal embedding, evaluated on
        # noisy embeddings (guarded here would just hide degradation behind p0)
        shots_res = []
        for sh in A["shots_sweep"]:
            Zsh = make_embeddings({**cfg["qrc"], "shots": sh}, self.seeds[:1],
                                  seq_all[te], mask_all[te])
            Zsh = self.red_qrc.transform(Zsh)
            p = fm_ideal_ug.predict(self.p0_all_band[te], Zsh, self.ctx_all[te])
            shots_res.append((sh, float(ev.auprc_fn(y_all[te], p))))
        ab["shots_sweep"] = shots_res

        sweep_q, sweep_k, sweep_dt = {}, {}, {}
        for nq in A["qubit_sweep"]:
            sweep_q[str(nq)], _ = self._band_auprc_for(
                {**cfg["qrc"], "n_qubits": nq, "input_qubits": min(3, nq - 1)})
        for kk in A["K_sweep"]:
            m1 = mask_all.copy()
            for i in range(len(m1)):
                idx = np.where(m1[i] > 0)[0]
                if len(idx) > kk:
                    m1[i, idx[:-kk]] = 0.0
            sweep_k[str(kk)], _ = self._band_auprc_for({**cfg["qrc"]}, seq_all, m1)
        for dtv in A.get("dt_sweep", []):
            sweep_dt[str(dtv)], _ = self._band_auprc_for({**cfg["qrc"], "dt": dtv})
        ab["qubit_sweep_band_auprc"] = sweep_q
        ab["K_sweep_band_auprc"] = sweep_k
        ab["dt_sweep_band_auprc"] = sweep_dt

    # ===================================================================
    # Stage F -- distillation to a real-time classical student
    # ===================================================================
    def distill_student(self):
        cfg, seed = self.cfg, self.seed
        yte, y_all = self.yte, self.y_all
        tr, va, te = self.tr, self.va, self.te
        self.log("distillation to a real-time classical student ...")
        # target = the hybrid's p_final on the train+val ambiguous band
        fm_q3 = self.fusion_models[self.q3_variant]
        _zs = self.q3_variant == "score_fusion"
        pf_tr = fm_q3.predict(self.p0_all_band[tr], self.Z_qrc[tr], self.ctx_all[tr],
                              self.z_score_all[tr] if _zs else None)
        pf_va = fm_q3.predict(self.p0_all_band[va], self.Z_qrc[va], self.ctx_all[va],
                              self.z_score_all[va] if _zs else None)
        Xband_tr = np.vstack([self.Xtr.to_numpy()[self.amb["train"]],
                              self.Xva.to_numpy()[self.amb["val"]]])
        yb_tr = np.concatenate([pf_tr, pf_va])
        Xband_te = self.Xte.to_numpy()[self.amb["test"]]
        p_student, lat_ms = train_student(Xband_tr, yb_tr, Xband_te, cfg["distill"], seed)
        student_band_auprc = float(ev.auprc_fn(y_all[te], p_student))
        self.p_student_e2e = self.reconstruct(p_student)
        self.metrics["operational"]["distilled_student"] = dict(
            band_auprc=student_band_auprc,
            hybrid_band_auprc=float(ev.auprc_fn(
                y_all[te], self.fusion_preds[self.q3_variant]["test"])),
            full_test_auprc=float(ev.auprc_fn(yte, self.p_student_e2e)),
            inference_latency_ms_per_txn=lat_ms,
        )
        self.log(f"  student band AUPRC {student_band_auprc:.4f} vs hybrid "
                 f"{ev.auprc_fn(y_all[te], self.fusion_preds[self.q3_variant]['test']):.4f}  "
                 f"latency {lat_ms:.4f} ms/txn")

    # ===================================================================
    # attribution + predictions.csv  (challenge outputs 1-3)
    # ===================================================================
    def write_predictions(self):
        test, yte = self.test, self.yte
        band, amb = self.band, self.amb
        self.log("attribution + predictions.csv ...")
        feat_names = list(self.Xte.columns)
        lgb_attr = attr.lightgbm_attribution(self.b4, self.Xte.to_numpy(), feat_names, top_k=5)
        fus_groups = attr.fusion_group_attribution(
            self.fusion_models[self.q3_variant], self.q3_variant,
            self.Z_qrc.shape[1], self.ctx_names)
        self.metrics["fusion_group_attribution"] = fus_groups

        route_label = np.full(len(test), "classical_approve", dtype=object)
        route_label[self.p0["test"] >= band["tau_high"]] = "classical_decline_or_review"
        route_label[amb["test"]] = "qrc_second_opinion"

        fraud_prob = self.p_q3_e2e
        fraud_pred = (fraud_prob >= self.t_hybrid).astype(int)

        band_pos = {int(t): i for i, t in enumerate(self.ids["test"])}
        rows = []
        for i, tid in enumerate(test["TransactionID"].to_numpy()):
            if amb["test"][i] and int(tid) in band_pos:
                a = {"classical_score_p0": round(float(self.p0["test"][i]), 4),
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
        pd.DataFrame(rows).to_csv(self.outdir / "predictions.csv", index=False)

    # ===================================================================
    # pre-registered success criteria (frozen before the test peek)
    # ===================================================================
    def evaluate_success_criteria(self):
        cfg = self.cfg
        m = self.metrics
        self.log("evaluating frozen success criteria ...")
        c1 = m["deltas"]["c1_auprc_Q3_vs_B4_full"]
        rc0 = cfg["routing"]["target_fraud_recall"]
        c2 = m["deltas"][f"c2_fdr_Q3_vs_B4_recall{rc0}"]
        c3 = m["deltas"]["c3_auprc_Q2_vs_ESN_band"]
        c1_pass = c1["delta"] > 0 and c1["ci_low"] > 0
        c2_pass = c2["delta"] < 0 and c2["ci_high"] < 0
        c3_beats = c3["delta"] > 0 and c3["ci_low"] > 0
        c3_parity = c3["ci_low"] > -0.02          # not meaningfully worse than the ESN
        c4_pass = all(
            (m["seeds"][str(s)]["delta_auprc_full"]["ci_low"] > 0 or
             m["seeds"][str(s)]["delta_fdr_full"]["ci_high"] < 0)
            for s in self.seeds)
        # A small (6-8 qubit) reservoir is, per Fujii & Nakajima, in the class of
        # a ~100-500 node ESN, so "beats the ESN" is a stretch goal; the honest
        # bar is PARITY with the classical reservoir while still improving on the GBDT.
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
        json.dump(sc, open(self.outdir / "success_criteria.json", "w"), indent=2)
        self.sc = sc

    # ===================================================================
    # manifest + metrics.json + report.md + plots
    # ===================================================================
    def finalize(self):
        cfg = self.cfg
        band = self.band
        yte = self.yte
        self.manifest["frozen"] = dict(
            entity_key="E2", split="chronological_60_20_20",
            tau_low=band["tau_low"], tau_high=band["tau_high"],
            t_star=band["t_star"], hybrid_decision_threshold=self.t_hybrid,
            target_fraud_recall=self.rc_t, routing_budget=cfg["routing"]["routing_budget"],
            reservoir_seeds=self.seeds,
        )
        self.manifest["quantum_resource"] = dict(
            n_qubits=cfg["qrc"]["n_qubits"],
            input_qubits=cfg["qrc"]["input_qubits"],
            reupload_rounds=cfg["qrc"]["reupload_rounds"],
            virtual_nodes=cfg["qrc"]["virtual_nodes"],
            evolution_steps_per_txn=cfg["sequence"]["K"] * cfg["qrc"]["reupload_rounds"]
            * cfg["qrc"]["virtual_nodes"],
            observables_per_node=QuantumReservoir(cfg["qrc"], self.seeds[0]).n_obs,
            embedding_dim=int(self.Z_qrc.shape[1]),
            spatial_multiplex=cfg["qrc"]["spatial_multiplex"],
            shots="ideal_statevector" if cfg["qrc"]["shots"] is None else cfg["qrc"]["shots"],
            simulator="numpy density-matrix (exact); PennyLane cross-check in src/qrc_pennylane.py",
            qrc_embed_seconds=self.qrc_secs,
        )
        self.manifest["runtime_seconds"] = time.time() - self.t0
        json.dump(self.manifest, open(self.outdir / "manifest.json", "w"),
                  indent=2, default=str)
        json.dump(self.metrics, open(self.outdir / "metrics.json", "w"),
                  indent=2, default=str)

        report.write_report(self.outdir, cfg, self.metrics, self.sc, self.manifest,
                            band, self.df, self.amb, yte)

        self._plots()

        self.log(f"DONE in {time.time() - self.t0:.1f}s -> {self.outdir}")
        print("\n==== SUCCESS CRITERIA ====")
        for k in ("c1_delta_auprc_positive_ci", "c2_false_decline_reduced_ci",
                  "c3_vs_esn_on_band", "c4_holds_across_seeds"):
            print(f"  {k:38s} {'PASS' if self.sc[k]['pass_'] else 'fail'}")
        print(f"  {'OVERALL measurable improvement':38s} "
              f"{'YES' if self.sc['overall_measurable_improvement'] else 'NO'}")

    def _plots(self):
        cfg = self.cfg
        yte = self.yte
        y_all, te = self.y_all, self.te
        outdir = self.outdir / "plots"
        try:
            self.log("plots ...")
            plt_.pr_curves({"Classic only (LightGBM)": (yte, self.p_b4_raw["test"]),
                            "Classic + classical reservoir (ESN)": (yte, self.p_esn_e2e),
                            "Classic + quantum reservoir (QRC)": (yte, self.p_q3_e2e)}, outdir)
            plt_.roc_curves({"Classic only (LightGBM)": (yte, self.p_b4_raw["test"]),
                             "Classic + quantum reservoir (QRC)": (yte, self.p_q3_e2e)}, outdir)
            plt_.calibration_plot({"Classic only (LightGBM), calibrated": (yte, self.p0["test"]),
                                   "Classic + quantum reservoir (QRC)": (yte, self.p_q3_e2e)}, outdir)
            plt_.routing_plot(self.p0["test"], yte, self.band, outdir)
            bm = self._band_models
            # plain-English display labels for the plot only -- the underlying
            # dict keys (B4_p0, Q0_qrc_only, ...) stay unchanged since report.py
            # and metrics.json look them up by those exact technical names.
            plot_label = {
                "B4_p0": "Classic only (LightGBM)",
                "Q0_qrc_only": "Reservoir only (QRC)",
                "Q1_score_fusion": "Fusion - score level",
                "Q2_feature_fusion": "Fusion - feature level",
                "Q2_no_p0_ablation": "Fusion - feature level, no classic (ablation)",
                f"Q_deployed[{self.q3_variant}]": "Deployed hybrid (classic + QRC)",
                f"ESN_deployed[{self.q3_variant}]": "Deployed hybrid (classic + ESN)",
            }
            plt_.bar_compare([plot_label.get(k, k) for k in bm.keys()],
                             [ev.auprc_fn(y_all[te], p) for p in bm.values()],
                             "AUPRC on the ambiguous band (test)", "AUPRC",
                             outdir, "band_auprc.png",
                             ref=ev.auprc_fn(y_all[te], self.p0["test"][self.amb["test"]]))
            if cfg["ablations"]["enabled"]:
                ab = self.metrics["ablations"]
                names = ["ideal"] + [k for k in ("no_entanglement", "random_reservoir",
                         "shuffle_sequence", "no_history_K1") if k in ab]
                vals = [ab["ideal_feature_fusion_band_auprc"]] + [ab[k] for k in names[1:]]
                ablation_label = {
                    "ideal": "Full model (no ablation)",
                    "no_entanglement": "No entanglement (ablation)",
                    "random_reservoir": "Random reservoir (ablation)",
                    "shuffle_sequence": "Shuffled time order (ablation)",
                    "no_history_K1": "No history, K=1 (ablation)",
                }
                plt_.bar_compare([ablation_label.get(n, n) for n in names], vals,
                                 "QRC ablations - band AUPRC", "AUPRC",
                                 outdir, "ablations.png",
                                 ref=ev.auprc_fn(y_all[te], self.p0["test"][self.amb["test"]]))
                if ab.get("shots_sweep"):
                    sh, au = zip(*ab["shots_sweep"])
                    plt_.shots_plot(list(sh), list(au), outdir,
                                    ideal=ab["ideal_feature_fusion_band_auprc"])
        except Exception as e:                       # pragma: no cover
            self.log(f"  plotting failed ({e!r}) - deliverables already written")
