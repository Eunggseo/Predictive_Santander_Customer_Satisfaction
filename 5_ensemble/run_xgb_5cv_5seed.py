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
OUTPUT_DIR = Path("xgb_5cv_5seed_outputs")


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

    train = pd.read_csv("train_clean.csv")
    test = pd.read_csv("test_clean.csv")

    target_col = "TARGET"
    id_col = "ID"

    feature_cols = [c for c in train.columns if c not in [id_col, target_col]]
    feature_cols = [c for c in feature_cols if c in test.columns]

    x = train[feature_cols]
    y = train[target_col].values
    x_test = test[feature_cols]

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Train matrix: {x.shape[0]} rows, {x.shape[1]} features")
    print(f"Test matrix: {x_test.shape[0]} rows, {x_test.shape[1]} features")
    print(f"Target rate: {y.mean():.6f}")
    print(f"Seeds: {SEEDS}")

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

    summary_path = OUTPUT_DIR / "xgb_5cv_5seed_auc_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    oof_path = OUTPUT_DIR / "oof_xgb_5cv_5seed.csv"
    oof_df = pd.DataFrame(
        {
            "ID": train[id_col],
            "TARGET": train[target_col],
            "oof_xgb_5cv_5seed": oof_sum,
            **seed_oof_columns,
        }
    )
    oof_df.to_csv(oof_path, index=False)

    submission_path = OUTPUT_DIR / "submission_xgb_5cv_5seed.csv"
    pd.DataFrame({"ID": test[id_col], "TARGET": test_sum}).to_csv(
        submission_path, index=False
    )

    detail_path = OUTPUT_DIR / "xgb_5cv_5seed_details.json"
    detail_path.write_text(
        json.dumps(
            {
                "seeds": SEEDS,
                "n_splits": N_SPLITS,
                "n_models": len(SEEDS) * N_SPLITS,
                "features": feature_cols,
                "overall_oof_auc": round(overall_auc, 6),
                "model_params": build_xgb(SEEDS[0]).get_params(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print(f"Repeated-seed XGBoost OOF AUC: {overall_auc:.6f}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {submission_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
