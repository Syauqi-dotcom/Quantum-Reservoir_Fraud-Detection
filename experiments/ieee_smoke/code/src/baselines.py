"""Classical baselines: Logistic Regression (B0) and LightGBM (B1 static / B4 +rolling)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:                                    # pragma: no cover
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:                                    # pragma: no cover
    HAS_CATBOOST = False


def train_logreg(X, y):
    m = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    m.fit(X, y)
    return m


def make_gbdt(cfg_lgb: dict, seed: int):
    """Unfitted GBDT estimator (used for cross_val_predict OOF scores too)."""
    p = dict(cfg_lgb)
    p.pop("calibration", None)
    if HAS_LGB:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=p.get("n_estimators", 600),
            learning_rate=p.get("learning_rate", 0.03),
            num_leaves=p.get("num_leaves", 64),
            subsample=p.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.8),
            min_child_samples=p.get("min_child_samples", 40),
            class_weight=p.get("class_weight", "balanced"),
            random_state=seed, n_jobs=4, verbose=-1,
        )
    return HistGradientBoostingClassifier(                     # pragma: no cover
        learning_rate=p.get("learning_rate", 0.05),
        max_iter=p.get("n_estimators", 400),
        max_leaf_nodes=p.get("num_leaves", 63),
        class_weight="balanced", random_state=seed,
    )


def train_gbdt(X, y, cfg_lgb: dict, seed: int):
    model = make_gbdt(cfg_lgb, seed)
    model.fit(X, y)
    return model


def train_catboost(X, y, cfg_cat: dict, seed: int):
    """CatBoost baseline - per the IEEE-CIS 1st-place writeup, the strongest
    SINGLE model of the three GBDTs they ensembled (0.9408 vs LightGBM's
    0.9384 and XGBoost's 0.9324 private-LB AUC), not just an ensemble filler."""
    if not HAS_CATBOOST:                              # pragma: no cover
        raise RuntimeError("catboost is not installed - pip install catboost")
    model = CatBoostClassifier(
        iterations=cfg_cat.get("iterations", 600),
        learning_rate=cfg_cat.get("learning_rate", 0.05),
        depth=cfg_cat.get("depth", 6),
        auto_class_weights="Balanced",
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        thread_count=4,
    )
    model.fit(np.asarray(X), np.asarray(y))
    return model


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]
