# Predictive_Santander_Customer_Satisfaction

## Kaggle Score Log

| Submission / Notebook | Private Score | Public Score | Notes |
|---|---:|---:|---|
| `XG boost basic model.ipynb` / Initial Submission_4.11.26 | 0.82347 | 0.83881 | Base XGBoost submission. |
| `XG boost wenyu-s-cleaned-data.ipynb` / Wenyu cleaned data Version 2 | 0.82349 | 0.83785 | XGBoost using Wenyu's cleaned data / EDA-cleaned data. |
| `submission_extratrees.csv` | 0.76378 | 0.78230 | Base ExtraTrees submission. |
| `submission_random_forest.csv` | 0.80502 | 0.81883 | Base Random Forest submission. |
| `submission_logistic_regression.csv` | 0.77733 | 0.79668 | Base Logistic Regression submission. |
| `submission_lightgbm.csv` | 0.82311 | 0.83706 | Base LightGBM submission. |
| `Model_LightGBM_FE.ipynb` / `submission_lightgbm_fe.csv` | 0.83777 | 0.82392 | LightGBM with zero/nonzero counts and `var38` features; OOF AUC 0.83825. |

## EDA v2 Summary

Notebook: `santander_eda_v2.ipynb`

Key findings:

| EDA question | Finding |
|---|---|
| Do dissatisfied and satisfied customers differ in sparsity? | Yes. Mean `nonzero_count` is lower for dissatisfied customers: TARGET=0 has 33.59 non-zero features on average, while TARGET=1 has 27.03. |
| Does activity bucket relate to dissatisfaction? | Yes. The lowest activity bucket has a 0.08952 dissatisfied rate, about 2.26x the overall 0.03957 rate. The highest activity bucket has a lower 0.02982 rate. |
| Is the `var38` peak value a special group? | Somewhat. The peak group is 19.56% of customers and has a 0.04130 dissatisfied rate versus 0.03915 for non-peak customers. |
| Do the FE features have standalone signal? | `zero_count` and `nonzero_count` have the strongest single-feature signal, with direction-adjusted AUC 0.63423. `var38_log` has weaker signal at 0.56959; `var38_is_peak` is near neutral at 0.50445. |

EDA-to-modeling decisions:

| EDA finding | Modeling decision | Result |
|---|---|---|
| Many features are sparse; row activity differs by customer group | Added `zero_count` / `nonzero_count` | Tested in LightGBM FE; OOF AUC improved from 0.83749 to 0.83825; Public/Private changed |
| `var38` has a known peak value that behaves like a special group | Added `var38_is_peak` / `var38_log` | Tested in LightGBM FE |
| TARGET is highly imbalanced, with overall dissatisfied rate near 3.96% | Next: test `scale_pos_weight` or `class_weight` | Pending |
