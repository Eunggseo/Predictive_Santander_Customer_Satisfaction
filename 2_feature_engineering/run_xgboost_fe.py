import warnings

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


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    warnings.filterwarnings("ignore", message=".*Parameters:.*use_label_encoder.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.training")

    train = pd.read_csv("train_clean.csv")
    test = pd.read_csv("test_clean.csv")

    train, test = add_fe(train, test)

    target_col = "TARGET"
    id_col = "ID"
    feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

    x = train[feature_cols]
    y = train[target_col]
    x_test = test[feature_cols]

    print("Train shape after FE:", x.shape)
    print("Test shape after FE:", x_test.shape)
    print("Train/test column diff:", set(x.columns) - set(x_test.columns), set(x_test.columns) - set(x.columns))
    print(x[["zero_count", "nonzero_count", "var38_log", "var38_is_peak"]].head())

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

        print(f"Fold {fold} AUC: {fold_auc:.5f}")

    oof_auc = roc_auc_score(y, oof_pred)

    print("=" * 40)
    print("Fold AUCs:", [round(score, 5) for score in fold_scores])
    print(f"OOF AUC: {oof_auc:.5f}")

    submission = pd.DataFrame({"ID": test["ID"], "TARGET": test_pred})
    submission.to_csv("submission_xgboost_fe.csv", index=False)
    print("Saved: submission_xgboost_fe.csv")

    result = pd.DataFrame(
        [
            {
                "Model": "XGBoost FE",
                "Private Score": "?",
                "Public Score": "?",
                "OOF AUC": round(oof_auc, 5),
                "Hyperparameters": "same as XGBoost cleaned_v1",
                "Notes": "Added zero_count, nonzero_count, var38_log, var38_is_peak",
            }
        ]
    )
    result.to_csv("xgboost_fe_results.csv", index=False)
    print("Saved: xgboost_fe_results.csv")


if __name__ == "__main__":
    main()
