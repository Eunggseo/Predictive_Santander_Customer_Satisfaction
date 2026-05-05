import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SEEDS = [42, 2024, 3407, 8803, 9187]
N_SPLITS = 5
OUTPUT_PATH = Path("basic_xgb_5fold_5seed.csv")
OOF_PATH = Path("oof_basic_xgb_5fold_5seed.csv")
SUMMARY_PATH = Path("basic_xgb_5fold_5seed_auc_summary.csv")
DETAIL_PATH = Path("basic_xgb_5fold_5seed_details.json")


def build_xgb(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def main() -> None:
    warnings.filterwarnings("ignore", message=".*Parameters:.*use_label_encoder.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.training")

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    target_col = "TARGET"
    id_col = "ID"

    feature_cols = [c for c in train.columns if c not in [id_col, target_col]]
    feature_cols = [c for c in feature_cols if c in test.columns]

    x = train[feature_cols]
    y = train[target_col].values
    x_test = test[feature_cols]

    print(f"Train matrix: {x.shape[0]} rows, {x.shape[1]} features")
    print(f"Test matrix: {x_test.shape[0]} rows, {x_test.shape[1]} features")
    print(f"Target rate: {y.mean():.6f}")
    print(f"Seeds: {SEEDS}")
    print("Input files: train.csv, test.csv")

    oof_sum = np.zeros(len(train))
    test_sum = np.zeros(len(test))
    seed_oof_columns = {}
    rows = []

    for seed_index, seed in enumerate(SEEDS, 1):
        print(f"\n[Seed {seed_index}/{len(SEEDS)}] random_state={seed}")

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(train))
        seed_test = np.zeros(len(test))

        for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y), 1):
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            model_seed = seed + fold
            model = build_xgb(model_seed)
            model.fit(x_train, y_train)

            valid_pred = model.predict_proba(x_valid)[:, 1]
            fold_auc = roc_auc_score(y_valid, valid_pred)
            seed_oof[valid_idx] = valid_pred
            seed_test += model.predict_proba(x_test)[:, 1] / N_SPLITS

            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "model_random_state": model_seed,
                    "fold_auc": round(fold_auc, 6),
                }
            )
            print(f"  Fold {fold} AUC: {fold_auc:.6f}")

        seed_auc = roc_auc_score(y, seed_oof)
        rows.append(
            {
                "seed": seed,
                "fold": "seed_oof",
                "model_random_state": "",
                "fold_auc": round(seed_auc, 6),
            }
        )
        print(f"[Seed OOF] AUC: {seed_auc:.6f}")

        seed_oof_columns[f"oof_seed_{seed}"] = seed_oof
        oof_sum += seed_oof / len(SEEDS)
        test_sum += seed_test / len(SEEDS)

    overall_auc = roc_auc_score(y, oof_sum)
    rows.append(
        {
            "seed": "all",
            "fold": "mean_oof",
            "model_random_state": "",
            "fold_auc": round(overall_auc, 6),
        }
    )

    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(
        {
            "ID": train[id_col],
            "TARGET": train[target_col],
            "oof_basic_xgb_5fold_5seed": oof_sum,
            **seed_oof_columns,
        }
    ).to_csv(OOF_PATH, index=False)
    pd.DataFrame({"ID": test[id_col], "TARGET": test_sum}).to_csv(
        OUTPUT_PATH, index=False
    )
    DETAIL_PATH.write_text(
        json.dumps(
            {
                "seeds": SEEDS,
                "n_splits": N_SPLITS,
                "n_models": len(SEEDS) * N_SPLITS,
                "input_train": "train.csv",
                "input_test": "test.csv",
                "features": feature_cols,
                "overall_oof_auc": round(overall_auc, 6),
                "model_params": build_xgb(SEEDS[0]).get_params(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print(f"Basic XGBoost 5-fold x 5-seed OOF AUC: {overall_auc:.6f}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {OOF_PATH}")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {DETAIL_PATH}")


if __name__ == "__main__":
    main()
