# Santander Customer Satisfaction

Predictive Analytics course project for the Kaggle Santander Customer
Satisfaction competition. The target is `TARGET=1`, dissatisfied customers, and
the evaluation metric is ROC AUC.

## Final Result

Final model: **Stacked Ensemble + Rank Average + 4-rule post-processing**

| Model | Owner | Private AUC | Public AUC | OOF AUC | Notes |
|---|---|---:|---:|---:|---|
| Final: Stacked Ensemble + Rank Avg + 4-rule | Wenyu | 0.82642 | n/a | n/a | Austin stacked ensemble rank-averaged with XGBoost FE + 4-rule, then final 4-rule |
| Stacked Ensemble | Austin | 0.82606 | 0.84001 | 0.84045 | XGBoost + LightGBM + CatBoost with logistic-regression stacking |
| XGBoost FE + 4-rule | Wenyu | 0.82568 | 0.83937 | 0.837924 | Activity/var38 FE plus validated 4-rule |
| LightGBM FE | Wenyu | 0.82392 | 0.83777 | 0.83825 | Activity/var38 FE |
| Base LightGBM | Hazel | 0.82311 | 0.83706 | 0.83749 | Baseline tree model |
| Random Forest | Hazel | 0.81883 | 0.80502 | 0.82093 | Baseline |
| Logistic Regression | Hazel | 0.79668 | 0.77733 | 0.79484 | Baseline |
| ExtraTrees | Hazel | 0.78230 | 0.76378 | 0.78118 | Baseline |

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

## Project Structure

| Path | Purpose |
|---|---|
| `1_eda/` | EDA v1 cleaning checks and EDA v2 modeling-driven analysis |
| `2_feature_engineering/` | Shared XGBoost feature-engineering helper |
| `3_model_exploration/` | Baseline and explored model families: LR, RF, ExtraTrees, LightGBM, XGBoost, NN |
| `4_imbalance_handling/` | LightGBM FE and `scale_pos_weight` experiments |
| `5_ensemble/` | Austin stacking notebook and rank/blend ensemble scripts |
| `6_post_processing/` | Final 4-rule validation and post-processing scripts |
| `7_feature_analysis/` | XGBoost gain feature importance and EDA/business figures |
| `8_final_pipeline/` | Final reproducible submission script |
| `Hazel_Feature_Analysis/` | Hazel's SHAP and business feature-analysis report |
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
- Feature analysis from Austin's three-model stack and Hazel's XGBoost SHAP
  both point to age, balance, activity/sparsity, and historical balance as
  central predictive signals.

## Data Note

The original Kaggle `train.csv` and `test.csv` are required to run the full
pipeline locally. They are not intended to be uploaded as project artifacts.
