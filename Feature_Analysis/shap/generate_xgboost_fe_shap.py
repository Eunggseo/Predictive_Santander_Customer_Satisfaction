import warnings

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import shap
from pandas.errors import PerformanceWarning
from xgboost import XGBClassifier

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

    # ---- compute SHAP values ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    # ---- mean |SHAP| table ----
    mean_abs_shap = pd.DataFrame(
        {"feature": feature_cols, "mean_abs_shap": abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    mean_abs_shap["shap_pct"] = (
        mean_abs_shap["mean_abs_shap"] / mean_abs_shap["mean_abs_shap"].sum()
    )

    top20 = mean_abs_shap.head(20).copy()
    top20.to_csv("xgboost_fe_top20_shap.csv", index=False)

    # ---- summary beeswarm plot (top 20) ----
    shap.summary_plot(
        shap_values, x, max_display=20, show=False, plot_size=(10, 7)
    )
    plt.title("Top 20 XGBoost FE SHAP Summary")
    plt.tight_layout()
    plt.savefig("xgboost_fe_top20_shap_summary.png", dpi=200)
    plt.close()

    # ---- bar plot (top 20 mean |SHAP|) ----
    plot_df = top20.sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2f6f6d")
    plt.title("Top 20 XGBoost FE SHAP (mean |SHAP|)")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("xgboost_fe_top20_shap_bar.png", dpi=200)
    plt.close()

    # ---- dependence plots for top 6 features ----
    top6 = top20["feature"].head(6).tolist()
    for feat in top6:
        shap.dependence_plot(
            feat, shap_values, x, show=False, interaction_index=None
        )
        plt.title(f"SHAP dependence: {feat}")
        plt.tight_layout()
        plt.savefig(f"xgboost_fe_shap_dependence_{feat}.png", dpi=200)
        plt.close()

    print("Saved: xgboost_fe_top20_shap.csv")
    print("Saved: xgboost_fe_top20_shap_summary.png")
    print("Saved: xgboost_fe_top20_shap_bar.png")
    for feat in top6:
        print(f"Saved: xgboost_fe_shap_dependence_{feat}.png")
    print(top20.to_string(index=False))


if __name__ == "__main__":
    main()
