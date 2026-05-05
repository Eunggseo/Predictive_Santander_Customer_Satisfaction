import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict


RNG_SEED = 42
WEIGHTS = [12, 24.27]
BASELINE_ROWS = [
    {
        "Model": "Base LightGBM",
        "FE": "No",
        "scale_pos_weight": 1,
        "OOF AUC": 0.83749,
        "Public": 0.83706,
        "Private": 0.82311,
        "Notes": "baseline",
    },
    {
        "Model": "LightGBM FE",
        "FE": "Yes",
        "scale_pos_weight": 1,
        "OOF AUC": 0.83825,
        "Public": 0.82392,
        "Private": 0.83777,
        "Notes": "activity + var38",
    },
]


def add_fe(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fe = train.copy()
    test_fe = test.copy()

    feature_cols = [c for c in train_fe.columns if c not in ["ID", "TARGET"]]

    train_fe["zero_count"] = (train_fe[feature_cols] == 0).sum(axis=1)
    test_fe["zero_count"] = (test_fe[feature_cols] == 0).sum(axis=1)

    train_fe["nonzero_count"] = (train_fe[feature_cols] != 0).sum(axis=1)
    test_fe["nonzero_count"] = (test_fe[feature_cols] != 0).sum(axis=1)

    peak = 117310.979016494
    train_fe["var38_is_peak"] = (train_fe["var38"] == peak).astype(int)
    test_fe["var38_is_peak"] = (test_fe["var38"] == peak).astype(int)

    train_fe["var38_log"] = np.log1p(train_fe["var38"])
    test_fe["var38_log"] = np.log1p(test_fe["var38"])
    train_fe.loc[train_fe["var38"] == peak, "var38_log"] = 0
    test_fe.loc[test_fe["var38"] == peak, "var38_log"] = 0

    return train_fe, test_fe


def main() -> None:
    np.random.seed(RNG_SEED)
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        category=UserWarning,
    )

    train = pd.read_csv("train_clean.csv")
    test = pd.read_csv("test_clean.csv")
    y = train["TARGET"].values
    test_id = test["ID"].values

    train_fe, test_fe = add_fe(train, test)
    x_df = train_fe.drop(columns=["ID", "TARGET"])
    x_test_df = test_fe.drop(columns=["ID"])

    common_cols = [c for c in x_df.columns if c in x_test_df.columns]
    x_df = x_df[common_cols]
    x_test_df = x_test_df[common_cols]

    x = x_df.values
    x_test = x_test_df.values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31],
    }

    results = list(BASELINE_ROWS)
    detail = []

    print(f"Train FE matrix: {x.shape[0]} rows, {x.shape[1]} features")
    print(f"Target rate: {y.mean():.6f}")

    for weight in WEIGHTS:
        print(f"\n[Run] scale_pos_weight={weight}")
        lgb_clf = LGBMClassifier(
            random_state=RNG_SEED,
            n_jobs=-1,
            verbose=-1,
            scale_pos_weight=weight,
        )
        gs = GridSearchCV(
            estimator=lgb_clf,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=-1,
        )
        gs.fit(x, y)
        best_model = gs.best_estimator_

        oof = cross_val_predict(best_model, x, y, cv=cv, method="predict_proba")[:, 1]
        oof_auc = roc_auc_score(y, oof)

        y_test_pred = best_model.predict_proba(x_test)[:, 1]
        weight_label = str(weight).replace(".", "p")
        submission_path = Path(f"submission_lightgbm_fe_spw_{weight_label}.csv")
        pd.DataFrame({"ID": test_id, "TARGET": y_test_pred}).to_csv(submission_path, index=False)

        row = {
            "Model": "LightGBM FE weighted",
            "FE": "Yes",
            "scale_pos_weight": weight,
            "OOF AUC": round(oof_auc, 5),
            "Public": "?",
            "Private": "?",
            "Notes": "medium imbalance" if weight == 12 else "full imbalance",
        }
        results.append(row)
        detail.append(
            {
                "scale_pos_weight": weight,
                "best_params": gs.best_params_,
                "gridsearch_auc": round(gs.best_score_, 5),
                "oof_auc": round(oof_auc, 5),
                "submission": str(submission_path),
            }
        )

        print(f"[CV] Best params: {gs.best_params_}")
        print(f"[CV] Best CV AUC: {gs.best_score_:.5f}")
        print(f"[OOF] AUC: {oof_auc:.5f}")
        print(f"Saved: {submission_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("lightgbm_fe_weighted_results.csv", index=False)
    Path("lightgbm_fe_weighted_details.json").write_text(
        json.dumps(detail, indent=2),
        encoding="utf-8",
    )

    print("\nSaved: lightgbm_fe_weighted_results.csv")
    print("Saved: lightgbm_fe_weighted_details.json")
    print(results_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
