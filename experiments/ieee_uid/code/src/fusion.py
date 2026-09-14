"""
Stage E - readout / fusion.

Regularised logistic regression (kept deliberately simple so the QRC
contribution is auditable via the coefficients).  C is chosen on the
validation ambiguous band.  Variants map to the challenge experiment matrix:

  qrc_only            -> Q0   readout( z_qrc )
  score_fusion        -> Q1   readout( [logit(p0), qrc_only_score] )
  feature_fusion      -> Q2/Q3 readout( [logit(p0), z_qrc, context] )
  feature_fusion_no_p0-> ablation: is p0 doing all the work?
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class FusionModel:
    def __init__(self, variant: str, C_grid, guard: bool = True):
        self.variant = variant
        self.C_grid = list(C_grid)
        self.guard = guard        # False for ablations: report the TRUE effect
                                  # of a destructive intervention, never masked
                                  # by silently falling back to p0.
        self.scaler = StandardScaler()
        self.clf = None
        self.C_ = None
        self._cal = None          # isotonic map back to band prevalence

    def _design(self, p0, z, ctx, z_score=None):
        cols = []
        if self.variant == "qrc_only":
            cols = [z]
        elif self.variant == "score_fusion":
            cols = [_logit(p0)[:, None], z_score[:, None]]
        elif self.variant == "feature_fusion":
            cols = [_logit(p0)[:, None], z, ctx]
        elif self.variant == "feature_fusion_no_p0":
            cols = [z, ctx]
        else:
            raise ValueError(self.variant)
        return np.hstack(cols)

    def fit(self, p0_tr, z_tr, ctx_tr, y_tr, p0_va, z_va, ctx_va, y_va,
            z_score_tr=None, z_score_va=None):
        Xtr = self._design(p0_tr, z_tr, ctx_tr, z_score_tr)
        Xva = self._design(p0_va, z_va, ctx_va, z_score_va)
        self.scaler.fit(Xtr)
        Xtr_s, Xva_s = self.scaler.transform(Xtr), self.scaler.transform(Xva)
        # L2 with standardised features shrinks a strong predictor (logit p0)
        # exactly as hard as the noisy reservoir dimensions; search both the
        # regularisation strength and class-weighting so the read-out can fall
        # back to "mostly p0" when the reservoir has little to add, instead of
        # under-performing p0 alone.
        best = None
        for cw in (None, "balanced"):
            for C in self.C_grid:
                clf = LogisticRegression(max_iter=3000, C=C, class_weight=cw)
                clf.fit(Xtr_s, y_tr)
                ap = average_precision_score(y_va, clf.predict_proba(Xva_s)[:, 1])
                if best is None or ap > best[0]:
                    best = (ap, C, cw, clf)
        self.C_ = best[1]
        self.class_weight_ = best[2]
        self.clf = best[3]
        # calibrate the (class-weighted) logistic output back to the band's
        # natural fraud prevalence so it can be spliced into p0 without a
        # scale jump that would wreck the full-population ranking.
        raw_va = self.clf.predict_proba(Xva_s)[:, 1]
        self._cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
        self._cal.fit(raw_va, y_va)
        # safety net: never deploy a read-out that ranks worse than p0 alone
        # on the validation band it will be spliced into.
        ap_fusion = best[0]
        ap_p0 = average_precision_score(y_va, p0_va)
        # qrc_only / feature_fusion_no_p0 are diagnostics of the raw reservoir
        # signal and must report their true standalone number; guard only the
        # heads that are actually deployable end-to-end (score_fusion / feature_fusion).
        diagnostic = self.variant in ("qrc_only", "feature_fusion_no_p0")
        self._fallback_to_p0 = (self.guard and not diagnostic
                                and ap_p0 > ap_fusion + 1e-9)
        return self

    def predict(self, p0, z, ctx, z_score=None):
        if getattr(self, "_fallback_to_p0", False):
            # decided on the validation band only: the fusion read-out did not
            # beat p0 alone there, so deploy p0 unchanged rather than add noise.
            return np.clip(np.asarray(p0, float), 1e-6, 1 - 1e-6)
        X = self.scaler.transform(self._design(p0, z, ctx, z_score))
        raw = self.clf.predict_proba(X)[:, 1]
        if self._cal is not None:
            return np.clip(self._cal.predict(raw), 1e-6, 1 - 1e-6)
        return raw

    def coefficients(self):
        return self.clf.coef_.ravel()


def qrc_only_score(z_tr, y_tr, z_va, y_va, z_eval, C_grid):
    """A 1-D QRC score used as the single extra feature in score_fusion (Q1)."""
    if len(np.unique(y_tr)) < 2:
        return (np.full(len(z_tr), 0.5), np.full(len(z_va), 0.5),
                np.full(len(z_eval), 0.5))
    sc = StandardScaler().fit(z_tr)
    best = None
    for C in C_grid:
        clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
        clf.fit(sc.transform(z_tr), y_tr)
        ap = average_precision_score(y_va, clf.predict_proba(sc.transform(z_va))[:, 1])
        if best is None or ap > best[0]:
            best = (ap, clf)
    clf = best[1]
    return (clf.predict_proba(sc.transform(z_tr))[:, 1],
            clf.predict_proba(sc.transform(z_va))[:, 1],
            clf.predict_proba(sc.transform(z_eval))[:, 1])
