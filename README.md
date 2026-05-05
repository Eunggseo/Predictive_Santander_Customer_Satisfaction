# Santander Customer Satisfaction

Predictive Analytics course project for the Kaggle Santander Customer
Satisfaction competition. The target is `TARGET=1`, dissatisfied customers, and
the evaluation metric is ROC AUC.

## Final Result

Final model: **Stacked Ensemble + Rank Average + 4-rule post-processing**

| Model | Private AUC | Public AUC | OOF AUC | Notes |
|---|---:|---:|---:|---|
| Final: Stacked Ensemble + Rank Avg + 4-rule | 0.82642 | 0.84030 | 0.84045 | OOF inherited from Austin's base stacking; no separate CV was run for the post-hoc rank averaging and 4-rule test-prediction transformation |
| Stacked Ensemble | 0.82606 | 0.84001 | 0.84045 | XGBoost + LightGBM + CatBoost with logistic-regression stacking |
| XGBoost FE + 4-rule | 0.82568 | 0.83937 | 0.837924 | Activity/var38 FE plus validated 4-rule |
| LightGBM FE | 0.82392 | 0.83777 | 0.83825 | Activity/var38 FE |
| Base LightGBM | 0.82311 | 0.83706 | 0.83749 | Baseline tree model |
| Random Forest | 0.81883 | 0.80502 | 0.82093 | Baseline |
| Logistic Regression | 0.79668 | 0.77733 | 0.79484 | Baseline |
| ExtraTrees | 0.78230 | 0.76378 | 0.78118 | Baseline |

## How To Reproduce Final Submission

Run from the repository root:

```bash
python 8_final_pipeline/final_submission.py
```

This script reads:

- `results/submissions/austin_stacked_ensemble_082606.csv`
- `results/submissions/xgboost_fe_4rule_postprocess.csv`
- `test.csv`

It writes:

- `results/submissions/sub4_rankavg_4rule.csv`
- `sub4_rankavg_4rule.csv`

The final file is reproduced by rank-averaging Austin's stacked ensemble with
the XGBoost FE + 4-rule submission, then applying the final validated 4-rule
low-risk post-processing.

## Directory Tree

```text
.
├── README.md
├── data/
│   └── README.md
├── 1_eda/
│   ├── eda_v1_cleaning.ipynb
│   └── eda_v2_modeling_driven.ipynb
├── 2_feature_engineering/
│   └── run_xgboost_fe.py
├── 3_model_exploration/
│   ├── Model_LogisticRegression.ipynb
│   ├── Model_RandomForest.ipynb
│   ├── Model_ExtraTrees.ipynb
│   ├── Model_LightGBM.ipynb
│   ├── Model_XGBoost_FE.ipynb
│   ├── XGBoost_basic_model.ipynb
│   ├── XGBoost_cleaned_data.ipynb
│   └── Neural_Network_best_model.ipynb
├── 4_imbalance_handling/
│   ├── Model_LightGBM_FE.ipynb
│   └── run_lightgbm_fe_weighted.py
├── 5_ensemble/
│   ├── stacking_ensemble.ipynb
│   ├── rank_average_diverse_models.py
│   ├── blend_current_best_with_xgb_5seed.py
│   ├── make_xgb_5cv_5seed_ensembles.py
│   ├── run_basic_xgb_5fold_5seed.py
│   └── run_xgb_5cv_5seed.py
├── 6_post_processing/
│   ├── validate_4rule_final.py
│   ├── run_xgboost_fe_raw_4rule.py
│   ├── run_lightgbm_basic_4rule.py
│   └── blend_best_with_basic_xgb5seed_4rule.py
├── 7_feature_analysis/
│   ├── generate_xgboost_fe_feature_importance.py
│   ├── generate_activity_risk_comparison.py
│   ├── insert_activity_risk_chart_into_docs.py
│   ├── xgboost_fe_top20_feature_importance.csv
│   └── xgboost_fe_top20_feature_importance.png
├── 8_final_pipeline/
│   └── final_submission.py
├── Feature_Analysis/
│   ├── README.md
│   ├── report/
│   ├── shap/
│   └── business_analysis/
└── results/
    ├── submissions/
    ├── oof_predictions/
    ├── archive_submissions/
    ├── experiment_logs/
    ├── figures/
    ├── reports/
    ├── presentation/
    └── kaggle_screenshot/
```

## Project Structure

| Path | Purpose |
|---|---|
| `1_eda/` | EDA v1 cleaning checks and EDA v2 modeling-driven analysis |
| `2_feature_engineering/` | Shared XGBoost feature-engineering helper |
| `3_model_exploration/` | Baseline and explored model families: LR, RF, ExtraTrees, LightGBM, XGBoost, NN |
| `4_imbalance_handling/` | LightGBM FE and `scale_pos_weight` experiments |
| `5_ensemble/` | Stacking notebook and rank/blend ensemble scripts |
| `6_post_processing/` | Final 4-rule validation and post-processing scripts |
| `7_feature_analysis/` | XGBoost gain feature importance and EDA/business figures |
| `8_final_pipeline/` | Final reproducible submission script |
| `Feature_Analysis/` | SHAP and business feature-analysis report |
| `results/submissions/` | Key final submissions used by the final pipeline |
| `results/oof_predictions/` | Key OOF prediction files for validation |
| `results/archive_submissions/` | Older experiment submissions kept for traceability |
| `results/experiment_logs/` | Experiment summaries, details JSON, and archived outputs |
| `results/figures/` | EDA and feature-analysis figures |
| `results/kaggle_screenshot/` | Placeholder for the best Kaggle leaderboard screenshot |

## Final 4-Rule Post-Processing

The final post-processing layer uses four independently validated low-risk
conditions:

```python
mask_4rule = (
    (df["num_aport_var13_hace3"] >= 6)
    | (df["num_meses_var13_largo_ult3"] >= 1)
    | (df["var15"] < 23)
    | (df["var36"] == 0)
)
```

Validation script:

```bash
python 6_post_processing/validate_4rule_final.py
```

Validation result:

- Train union: 1,807 customers
- TARGET=1 within union: 0 customers
- XGBoost FE OOF AUC: 0.8376387514 -> 0.8379239075
- OOF AUC delta: +0.0002851561
- Decision: included in final pipeline

## Key Findings

- Lowest-activity customers show a 2.26x dissatisfaction lift versus the
  overall baseline.
- Zero account balance and no core product holding converge on nearly the same
  high-risk segment as the low-activity EDA cut.
- Feature analysis from the three-model stack and XGBoost SHAP both point to
  age, balance, activity/sparsity, and historical balance as central predictive
  signals.

## Data Note

The original Kaggle `train.csv` and `test.csv` are required to run the full
pipeline locally. They are not intended to be uploaded as project artifacts.
