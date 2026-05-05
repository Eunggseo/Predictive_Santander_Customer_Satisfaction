"""
4-rule post-processing - Final Validation
=========================================

Confirms the final four rules are zero-positive on the training set and
improve XGBoost FE OOF AUC. Re-validated: 2026-05.
"""

from pathlib import Path

import pandas as pd


TRAIN_PATH = Path("train.csv")
OOF_PATH = Path("results/oof_predictions/oof_xgboost_fe_4rule_check.csv")


def roc_auc_score_fallback(y_true: pd.Series, score: pd.Series) -> float:
    """Rank-based ROC AUC fallback for environments without scikit-learn."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    score = pd.Series(score).reset_index(drop=True)
    ranks = score.rank(method="average")
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    sum_pos_ranks = ranks[y_true == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = roc_auc_score_fallback


def final_4rule_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["num_aport_var13_hace3"] >= 6)
        | (df["num_meses_var13_largo_ult3"] >= 1)
        | (df["var15"] < 23)
        | (df["var36"] == 0)
    )


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    oof = pd.read_csv(OOF_PATH)

    print(f"Overall TARGET rate: {train['TARGET'].mean():.16f}")
    print()

    rules = {
        "num_aport_var13_hace3 >= 6": train["num_aport_var13_hace3"] >= 6,
        "num_meses_var13_largo_ult3 >= 1": train[
            "num_meses_var13_largo_ult3"
        ]
        >= 1,
        "var15 < 23": train["var15"] < 23,
        "var36 == 0": train["var36"] == 0,
    }

    for name, mask in rules.items():
        subset = train[mask]
        print(
            f"{name:<40} n={len(subset):<6} "
            f"TARGET=1={int(subset['TARGET'].sum()):<4} "
            f"rate={subset['TARGET'].mean():.1f}"
        )

    union = final_4rule_mask(train)
    print(
        f"\n4-rule union n={int(union.sum())}, "
        f"TARGET=1={int(train.loc[union, 'TARGET'].sum())}, "
        f"rate={train.loc[union, 'TARGET'].mean():.1f}"
    )

    raw_col = "oof_xgboost_fe"
    pp_col = "oof_xgboost_fe_4rule"
    if pp_col not in oof.columns:
        postprocessed = oof[raw_col].copy()
        postprocessed.loc[union.values] = 0.0
        oof[pp_col] = postprocessed

    raw_auc = roc_auc_score(oof["TARGET"], oof[raw_col])
    pp_auc = roc_auc_score(oof["TARGET"], oof[pp_col])
    print(f"\nRaw OOF AUC:         {raw_auc:.10f}")
    print(f"Postprocess OOF AUC: {pp_auc:.10f}")
    print(f"Delta:               +{pp_auc - raw_auc:.10f}")


if __name__ == "__main__":
    main()


# Expected output:
# Overall TARGET rate: 0.0395685345961589
# num_aport_var13_hace3 >= 6             n=141    TARGET=1=0    rate=0.0
# num_meses_var13_largo_ult3 >= 1        n=543    TARGET=1=0    rate=0.0
# var15 < 23                             n=1212   TARGET=1=0    rate=0.0
# var36 == 0                             n=411    TARGET=1=0    rate=0.0
# 4-rule union n=1807, TARGET=1=0, rate=0.0
# Raw OOF AUC:         0.8376387514
# Postprocess OOF AUC: 0.8379239075
# Delta:               +0.0002851561
