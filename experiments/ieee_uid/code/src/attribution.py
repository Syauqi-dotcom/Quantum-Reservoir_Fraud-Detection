"""
Feature attribution (challenge output #3).

  * LightGBM   : SHAP if available, else gain importance + per-row signed
                 contribution approximation.
  * Fusion     : standardised logistic coefficients -> grouped contribution
                 (classical score p0 vs QRC-observable groups vs context),
                 as the review recommends (no single observable is claimed
                 to "cause" fraud).
"""
from __future__ import annotations

import numpy as np

try:
    import shap
    HAS_SHAP = True
except Exception:                       # pragma: no cover
    HAS_SHAP = False


def lightgbm_attribution(model, X, feature_names, top_k=5):
    n = X.shape[0]
    contribs = None
    if HAS_SHAP:
        try:
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            contribs = np.asarray(sv)
        except Exception:
            contribs = None
    if contribs is None:
        try:
            booster = model.booster_
            raw = booster.predict(np.asarray(X), pred_contrib=True)
            contribs = np.asarray(raw)[:, :-1]
        except Exception:
            imp = getattr(model, "feature_importances_", np.ones(len(feature_names)))
            imp = imp / (imp.sum() + 1e-9)
            Xn = (np.asarray(X) - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-9)
            contribs = Xn * imp

    out = []
    for i in range(n):
        row = contribs[i]
        order = np.argsort(-np.abs(row))[:top_k]
        out.append([{"feature": str(feature_names[j]),
                     "contribution": round(float(row[j]), 4)} for j in order])
    return out


def fusion_group_attribution(fusion_model, variant, n_qrc, ctx_names):
    coef = fusion_model.coefficients()
    groups = {}
    idx = 0
    if variant in ("feature_fusion", "score_fusion"):
        groups["classical_score_p0"] = float(abs(coef[idx])); idx += 1
    if variant == "score_fusion":
        groups["qrc_score"] = float(abs(coef[idx])); idx += 1
        total = sum(groups.values()) + 1e-9
        return {k: round(v / total, 4) for k, v in groups.items()}
    if variant in ("qrc_only", "feature_fusion", "feature_fusion_no_p0"):
        groups["qrc_observables"] = float(np.sum(np.abs(coef[idx:idx + n_qrc])))
        idx += n_qrc
    for c in ctx_names:
        if idx < len(coef):
            groups[f"ctx:{c}"] = float(abs(coef[idx])); idx += 1
    total = sum(groups.values()) + 1e-9
    return {k: round(v / total, 4) for k, v in groups.items()}
