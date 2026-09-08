"""Classical baselines for the Vesta/IEEE-CIS Kaggle fraud dataset.

The competition files are rule-gated on Kaggle. This script retrieves OpenML
dataset 46858, whose metadata points to the original Kaggle competition, and
uses a deterministic 100,000-row stratified sample for a reproducible local
benchmark of the same four models used in the earlier experiments.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


SEED = 42
SAMPLE_ROWS = 100_000
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "work" / "openml"
SHARED_CACHE = ROOT.parents[0] / "work" / "openml"


def load_data() -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    from sklearn.datasets import fetch_openml

    cache = SHARED_CACHE if SHARED_CACHE.exists() else CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(data_id=46858, as_frame=True, data_home=cache)
    X_full = dataset.data.copy()
    y_full = pd.to_numeric(dataset.target, errors="raise").astype(int)

    if len(X_full) != 590_540 or int(y_full.sum()) != 20_663:
        raise ValueError(
            "OpenML mirror failed validation: expected 590,540 rows and "
            "20,663 fraud cases"
        )

    # IDs are not predictive features and can encourage memorization.
    id_columns = [c for c in X_full.columns if c.lower() == "transactionid"]
    X_full = X_full.drop(columns=id_columns)
    all_missing = X_full.columns[X_full.isna().all()].tolist()
    X_full = X_full.drop(columns=all_missing)

    sample_indices, _ = train_test_split(
        np.arange(len(y_full)),
        train_size=SAMPLE_ROWS,
        stratify=y_full,
        random_state=SEED,
    )
    X = X_full.iloc[sample_indices].reset_index(drop=True)
    y = y_full.iloc[sample_indices].reset_index(drop=True)
    audit = {
        "source_rows": len(X_full),
        "source_predictors_after_filtering": X_full.shape[1],
        "source_frauds": int(y_full.sum()),
        "sample_rows": len(X),
        "sample_predictors": X.shape[1],
        "sample_frauds": int(y.sum()),
        "dropped_id_columns": len(id_columns),
        "dropped_all_missing_columns": len(all_missing),
    }
    return X, y, audit


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


def models(positive_weight: float) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            scale_pos_weight=positive_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=24,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1,
            random_state=SEED,
        ),
    }


def evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    representation: str,
    pca_components: int | None = None,
) -> list[dict[str, object]]:
    negative, positive = np.bincount(y_train)
    rows: list[dict[str, object]] = []
    for name, model in models(negative / positive).items():
        steps: list[tuple[str, object]] = [("prep", make_preprocessor(X_train))]
        if pca_components is not None:
            steps.append(
                ("pca", PCA(n_components=pca_components, svd_solver="randomized", random_state=SEED))
            )
        steps.append(("model", model))
        pipeline = Pipeline(steps)

        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probability = pipeline.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        inference_ms_per_row = (time.perf_counter() - started) * 1000 / len(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        rows.append(
            {
                "representation": representation,
                "model": name,
                "accuracy": accuracy_score(y_test, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                "precision": precision_score(y_test, prediction, zero_division=0),
                "recall": recall_score(y_test, prediction, zero_division=0),
                "f1": f1_score(y_test, prediction, zero_division=0),
                "mcc": matthews_corrcoef(y_test, prediction),
                "roc_auc": roc_auc_score(y_test, probability),
                "pr_auc": average_precision_score(y_test, probability),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "training_seconds": training_seconds,
                "inference_ms_per_row": inference_ms_per_row,
            }
        )
    return rows


def main() -> None:
    X, y, audit = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    results = evaluate(X_train, X_test, y_train, y_test, "all retained features")
    results += evaluate(X_train, X_test, y_train, y_test, "PCA-8", 8)
    frame = pd.DataFrame(results)
    frame.to_csv(ROOT / "vesta_ieee_cis_results.csv", index=False)

    metadata = {
        "requested_dataset": "Kaggle IEEE-CIS Fraud Detection (Vesta Corporation)",
        "retrieval": "OpenML public mirror, dataset ID 46858, version 2",
        "kaggle_url": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        **audit,
        "sample_fraud_rate": float(y.mean()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_fraud_cases": int(y_test.sum()),
        "split": "stratified 80/20 holdout after deterministic stratified sampling",
        "random_seed": SEED,
        "decision_threshold": 0.5,
        "primary_metric": "PR-AUC (average precision)",
    }
    (ROOT / "vesta_ieee_cis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
