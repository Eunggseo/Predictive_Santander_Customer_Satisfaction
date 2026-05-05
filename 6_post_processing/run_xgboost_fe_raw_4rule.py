import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


RNG_SEED = 42


def add_fe(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = "TARGET"
    id_col = "ID"

    base_feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

    train["zero_count"] = (train[base_feature_cols] == 0).sum(axis=1)
    test["zero_count"] = (test[base_feature_cols] == 0).sum(axis=1)

    train["nonzero_count"] = (train[base_feature_cols] != 0).sum(axis=1)
    test["nonzero_count"] = (test[base_feature_cols] != 0).sum(axis=1)

    peak = 117310.979016494
    if "var38" in train.columns:
        train["var38_is_peak"] = (train["var38"] == peak).astype(int)
        test["var38_is_peak"] = (test["var38"] == peak).astype(int)

        train["var38_log"] = np.log1p(train["var38"])
        test["var38_log"] = np.log1p(test["var38"])

        train.loc[train["var38"] == peak, "var38_log"] = 0
        test.loc[test["var38"] == peak, "var38_log"] = 0

    return train, test


def four_rule_low_risk(df: pd.DataFrame) -> pd.Series:
    # ============================================================
    # 4-rule post-processing - FINAL VALIDATED VERSION
    # Re-validated: 2026-05
    # Train union: n=1807, TARGET rate=0.0% (zero-positive)
    # OOF AUC: 0.837639 -> 0.837924 (+0.000285)
    # Early exploration rules (var15<=22, num_aport>4.5, etc.)
    # are superseded by this business-readable version.
    # ============================================================
    return (
        (df["num_aport_var13_hace3"] >= 6)
        | (df["num_meses_var13_largo_ult3"] >= 1)
        | (df["var15"] < 23)
        | (df["var36"] == 0)
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    warnings.filterwarnings("ignore", message=".*Parameters:.*use_label_encoder.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.training")

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train_raw = train.copy()
    test_raw = test.copy()
    train, test = add_fe(train, test)

    target_col = "TARGET"
    id_col = "ID"
    feature_cols = [c for c in train.columns if c not in [id_col, target_col]]
    feature_cols = [c for c in feature_cols if c in test.columns]

    x = train[feature_cols]
    y = train[target_col]
    x_test = test[feature_cols]

    print("Input files: train.csv, test.csv")
    print("Train shape after FE:", x.shape)
    print("Test shape after FE:", x_test.shape)
    print(
        x[["zero_count", "nonzero_count", "var38_log", "var38_is_peak"]].head()
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), 1):
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RNG_SEED + fold,
            n_jobs=-1,
        )

        model.fit(x_train, y_train)

        valid_pred = model.predict_proba(x_valid)[:, 1]
        oof_pred[valid_idx] = valid_pred

        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(fold_auc)
        test_pred += model.predict_proba(x_test)[:, 1] / skf.n_splits

        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    oof_auc = roc_auc_score(y, oof_pred)

    train_rule = four_rule_low_risk(train_raw)
    test_rule = four_rule_low_risk(test_raw)
    oof_post = oof_pred.copy()
    test_post = test_pred.copy()
    oof_post[train_rule.values] = 0.0
    test_post[test_rule.values] = 0.0
    oof_post_auc = roc_auc_score(y, oof_post)

    submission_path = Path("submission_xgboost_fe_raw.csv")
    submission_4rule_path = Path("submission_xgboost_fe_raw_4rule_postprocess.csv")
    oof_path = Path("oof_xgboost_fe_raw_4rule_check.csv")
    result_path = Path("xgboost_fe_raw_4rule_results.csv")
    detail_path = Path("xgboost_fe_raw_4rule_details.json")

    pd.DataFrame({"ID": test["ID"], "TARGET": test_pred}).to_csv(
        submission_path, index=False
    )
    pd.DataFrame({"ID": test["ID"], "TARGET": test_post}).to_csv(
        submission_4rule_path, index=False
    )
    pd.DataFrame(
        {
            "ID": train["ID"],
            "TARGET": y,
            "oof_xgboost_fe_raw": oof_pred,
            "oof_xgboost_fe_raw_4rule": oof_post,
            "rule_4_low_risk": train_rule.astype(int),
        }
    ).to_csv(oof_path, index=False)

    result = {
        "Model": "XGBoost FE raw + 4-rule postprocess",
        "Private Score": "?",
        "Public Score": "?",
        "OOF AUC before": round(oof_auc, 6),
        "OOF AUC after": round(oof_post_auc, 6),
        "OOF delta": round(oof_post_auc - oof_auc, 6),
        "train_rule_n": int(train_rule.sum()),
        "train_rule_target_sum": int(train_raw.loc[train_rule, "TARGET"].sum()),
        "test_rule_n": int(test_rule.sum()),
        "submission": str(submission_4rule_path),
        "Notes": "Raw train/test with zero_count, nonzero_count, var38_log, var38_is_peak; set 4-rule low-risk predictions to 0.",
    }
    pd.DataFrame([result]).to_csv(result_path, index=False)
    detail_path.write_text(
        json.dumps(
            {
                "input_train": "train.csv",
                "input_test": "test.csv",
                "n_splits": skf.n_splits,
                "fold_aucs": [round(score, 6) for score in fold_scores],
                "oof_auc_before": round(oof_auc, 6),
                "oof_auc_after": round(oof_post_auc, 6),
                "oof_delta": round(oof_post_auc - oof_auc, 6),
                "train_rule_n": int(train_rule.sum()),
                "train_rule_target_sum": int(train_raw.loc[train_rule, "TARGET"].sum()),
                "test_rule_n": int(test_rule.sum()),
                "features": feature_cols,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 50)
    print("Fold AUCs:", [round(score, 6) for score in fold_scores])
    print(f"OOF AUC before 4-rule: {oof_auc:.6f}")
    print(f"OOF AUC after 4-rule: {oof_post_auc:.6f}")
    print(f"OOF delta: {oof_post_auc - oof_auc:+.6f}")
    print(
        "Rule counts:",
        {
            "train_n": int(train_rule.sum()),
            "train_target_sum": int(train_raw.loc[train_rule, "TARGET"].sum()),
            "test_n": int(test_rule.sum()),
        },
    )
    print(f"Saved: {submission_path}")
    print(f"Saved: {submission_4rule_path}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {result_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
