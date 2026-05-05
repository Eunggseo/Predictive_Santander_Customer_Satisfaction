import warnings

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import PerformanceWarning
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2_feature_engineering"))
OUT_DIR = Path(__file__).resolve().parent

from run_xgboost_fe import RNG_SEED, add_fe


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    warnings.filterwarnings("ignore", message=".*Parameters:.*use_label_encoder.*")

    train = pd.read_csv("train_clean.csv")
    test = pd.read_csv("test_clean.csv")
    train, test = add_fe(train, test)

    target_col = "TARGET"
    id_col = "ID"
    feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

    x = train[feature_cols]
    y = train[target_col]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    model.fit(x, y)

    gain_scores = model.get_booster().get_score(importance_type="gain")
    importance = (
        pd.Series(gain_scores, name="gain")
        .rename_axis("feature")
        .reset_index()
        .sort_values("gain", ascending=False)
    )
    importance["gain_pct"] = importance["gain"] / importance["gain"].sum()

    top20 = importance.head(20).copy()
    csv_path = OUT_DIR / "xgboost_fe_top20_feature_importance.csv"
    png_path = OUT_DIR / "xgboost_fe_top20_feature_importance.png"
    top20.to_csv(csv_path, index=False)

    plot_df = top20.sort_values("gain", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["gain"], color="#2f6f6d")
    plt.title("Top 20 XGBoost FE Feature Importance")
    plt.xlabel("Average gain")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(top20.to_string(index=False))


if __name__ == "__main__":
    main()
