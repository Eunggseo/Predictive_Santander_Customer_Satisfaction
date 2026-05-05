# Feature Importance & Business Insights

This folder contains the feature-analysis deliverables built from the XGBoost
FE model, including SHAP analysis, business-friendly feature bucketing, and the
final report assets.

## Folder Overview

```text
Feature_Analysis/
├── README.md
├── report/
│   ├── feature_importance_business_insights_report.pdf
├── shap/
│   ├── generate_xgboost_fe_shap.py
│   ├── xgboost_fe_top20_shap.csv
│   ├── xgboost_fe_top20_shap_summary.png
│   ├── xgboost_fe_top20_shap_bar.png
│   └── xgboost_fe_shap_dependence_*.png
└── business_analysis/
    ├── generate_xgboost_fe_business_analysis.py
    ├── business_top_features_summary.csv
    ├── business_var15_age.png
    ├── business_saldo_var30_binary.png
    ├── business_saldo_var30_quintile.png
    ├── business_ind_var30.png
    └── business_ind_var26_cte.png
```

## Key Takeaways

1. SHAP and gain-based importance tell different stories. SHAP ranks
   `var15` first, accounting for 30.7% of total SHAP importance, while
   `ind_var30` falls to rank 17. Because gain can overstate binary indicators,
   the presentation should use SHAP for the feature story.

2. The largest drivers are age, current balance, and historical balance. The
   top three SHAP features account for roughly half of total prediction signal.

3. `ind_var26_cte` identifies a small but higher-risk subgroup: 2.7% of
   customers with a 6.68% dissatisfaction rate, about 1.7x the baseline.

## How To Re-run

Both scripts use the shared XGBoost FE helper:

```python
from run_xgboost_fe import RNG_SEED, add_fe
```

To re-run locally:

1. Put `run_xgboost_fe.py`, `train_clean.csv`, and `test_clean.csv` next to the
   script or update the import/data paths.
2. Install SHAP for the SHAP script: `pip install shap`.
3. Run:

```bash
python shap/generate_xgboost_fe_shap.py
python business_analysis/generate_xgboost_fe_business_analysis.py
```

## Caveat

The `var15 < 23` bucket has a 0% dissatisfaction rate in the training data, but
this should not be interpreted as "all young customers are satisfied." The
group is very small and likely contains minor, secondary, or inactive accounts.
Use the 23-30 bucket as the more realistic young-customer baseline.
