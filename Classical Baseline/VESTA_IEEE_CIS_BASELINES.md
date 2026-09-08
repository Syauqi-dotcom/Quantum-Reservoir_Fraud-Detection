# Classical Baselines — Vesta / IEEE-CIS Fraud Detection

## Dataset and experimental protocol

Requested source: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data), with real e-commerce transactions supplied by Vesta Corporation. The competition files are gated by Kaggle sign-in and rule acceptance, so the experiment retrieved OpenML dataset 46858 version 2. Its metadata identifies the original Kaggle competition as the source.

The mirror contains 590,540 labeled transactions and 20,663 fraud cases (3.499%). Transaction and identity information are already merged. After removing 23 entirely missing columns, 432 predictors remain. For a locally reproducible comparison across all four models and PCA, a deterministic stratified sample of 100,000 rows was drawn, retaining 3,499 fraud cases and the original 3.499% fraud prevalence.

Protocol:

- Stratified 80/20 holdout after deterministic stratified sampling, seed 42
- Training: 80,000 rows; test: 20,000 rows with 700 fraud cases
- Median imputation for numerical features
- Most-frequent imputation and ordinal encoding for categorical features
- Standardization fitted only on the training partition
- Class-weighted logistic regression, Random Forest, XGBoost, and LightGBM
- Fixed decision threshold of 0.5
- All 432 retained predictors and a circuit-compatible PCA-8 representation
- Primary model-selection metric: PR-AUC (average precision)

## Measured holdout results

| Representation | Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC | FP | FN | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All 432 | Logistic regression | 83.45% | 77.04% | 13.67% | 70.14% | 22.88% | 0.2589 | 84.62% | 39.41% | 3,101 | 209 | 21.62 |
| All 432 | Random Forest | **97.44%** | 66.60% | **83.57%** | 33.43% | **47.76%** | **0.5192** | 89.87% | 55.02% | **46** | 466 | 39.92 |
| All 432 | XGBoost | 89.81% | **83.36%** | 22.21% | **76.43%** | 34.42% | 0.3767 | 91.12% | 56.51% | 1,874 | **165** | 29.03 |
| All 432 | LightGBM | 92.00% | 83.26% | 26.72% | 73.86% | 39.24% | 0.4135 | **91.54%** | **59.94%** | 1,418 | 183 | 21.88 |
| PCA-8 | Logistic regression | 83.29% | 70.55% | 11.57% | 56.86% | 19.23% | 0.2002 | 78.86% | 20.78% | 3,041 | 302 | 11.14 |
| PCA-8 | Random Forest | **97.10%** | 62.29% | **76.32%** | 24.86% | **37.50%** | **0.4255** | **83.22%** | **40.84%** | **54** | 526 | 29.00 |
| PCA-8 | XGBoost | 81.00% | 74.19% | 11.60% | **66.86%** | 19.76% | 0.2215 | 82.63% | 35.55% | 3,568 | **232** | 13.41 |
| PCA-8 | LightGBM | 84.40% | **74.64%** | 13.53% | 64.14% | 22.34% | 0.2434 | 82.88% | 36.28% | 2,870 | 251 | 13.12 |

## Interpretation

- **Best ranking model: LightGBM.** It obtains the highest PR-AUC (59.94%) and ROC-AUC (91.54%). It identifies 517 of 700 fraud cases at the fixed threshold, with 1,418 false alerts.
- **Best recall: XGBoost.** It finds 535 of 700 frauds (76.43%) but generates 1,874 false positives. Threshold tuning should determine whether this higher detection rate justifies the additional review burden.
- **Best fixed-threshold precision and F1: Random Forest.** Its 83.57% precision, 47.76% F1, and 97.44% accuracy appear strong, but it detects only 33.43% of fraud cases. It is suitable only when false-positive cost dominates missed-fraud cost.
- **The PCA-8 bottleneck is material.** The best PR-AUC declines from 59.94% to 40.84% (−19.10 percentage points), and the best ROC-AUC declines from 91.54% to 83.22% (−8.32 points). A quantum model using eight encoded dimensions must therefore be compared against the PCA-8 classical baselines on the identical sample and split.
- **Accuracy is not sufficient.** A majority-class predictor would already achieve approximately 96.50% accuracy while detecting no fraud. PR-AUC, recall at a fixed false-positive/review budget, and expected financial cost should drive deployment decisions.

## Finance application guidance

For Vesta-style card-not-present fraud screening, use the boosted tree as a real-time ranker and select the operating threshold according to fraud loss, manual-review capacity, customer friction, and transaction value. A practical hybrid design sends only uncertain or high-value cases to a compact quantum reservoir or feature map. The quantum stage should be retained only if it improves out-of-time PR-AUC, recall at fixed review volume, probability calibration, or expected monetary loss over the same-feature classical comparator.

For final competition-grade evaluation, replace random holdout with a forward-in-time split based on `TransactionDT`, report repeated temporal folds, tune the threshold using validation data only, and keep the final test prevalence unchanged. Identity availability should be treated explicitly because many transactions lack corresponding identity data.

## Reproducibility limitation

These are measured local benchmark results, but they use a 100,000-row stratified sample rather than all 590,540 rows. The sampling decision is recorded in the metadata and source code. Results should not be presented as Kaggle leaderboard scores; the original Kaggle test labels are unavailable.

