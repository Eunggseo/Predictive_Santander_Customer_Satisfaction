import warnings

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from run_xgboost_fe import add_fe


def bucket_stats(df: pd.DataFrame, bucket_col: str, target_col: str) -> pd.DataFrame:
    grouped = df.groupby(bucket_col, observed=True, dropna=False)[target_col]
    stats = grouped.agg(["count", "sum", "mean"]).reset_index()
    stats.columns = [bucket_col, "sample_size", "n_unsatisfied", "target_rate"]
    overall = df[target_col].mean()
    stats["lift"] = stats["target_rate"] / overall
    return stats


def plot_bucket(
    stats: pd.DataFrame,
    bucket_col: str,
    title: str,
    overall_rate: float,
    out_path: str,
    show_sample_size: bool = True,
) -> None:
    labels = stats[bucket_col].astype(str).tolist()
    sample_sizes = stats["sample_size"].tolist()
    target_rates = stats["target_rate"].tolist()

    if show_sample_size:
        fig, ax_rate = plt.subplots(figsize=(10, 6))
        ax_size = ax_rate.twinx()

        # background bars: sample size (light)
        ax_size.bar(
            labels, sample_sizes, color="#cfe0df", label="Sample size", zorder=1
        )
        ax_size.set_ylabel("Sample size")

        # foreground bars: target rate (dark, narrower)
        ax_rate.bar(
            labels,
            target_rates,
            color="#2f6f6d",
            width=0.5,
            label="Unsatisfied rate",
            zorder=3,
        )
        ax_rate.set_ylabel("Unsatisfied rate")
        ax_rate.set_ylim(0, max(target_rates) * 1.25)

        # baseline line
        ax_rate.axhline(
            overall_rate,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Overall ({overall_rate:.2%})",
            zorder=4,
        )

        # rate labels above dark bars
        for i, rate in enumerate(target_rates):
            ax_rate.text(
                i,
                rate,
                f"{rate:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#2f6f6d",
                fontweight="bold",
            )

        # put rate axis in front
        ax_rate.set_zorder(ax_size.get_zorder() + 1)
        ax_rate.patch.set_visible(False)

        # combined legend
        lines1, labels1 = ax_rate.get_legend_handles_labels()
        lines2, labels2 = ax_size.get_legend_handles_labels()
        ax_rate.legend(lines2 + lines1, labels2 + labels1, loc="upper right")
    else:
        fig, ax_rate = plt.subplots(figsize=(8, 6))
        ax_rate.bar(labels, target_rates, color="#2f6f6d", width=0.5)
        ax_rate.set_ylabel("Unsatisfied rate")
        ax_rate.set_ylim(0, max(target_rates) * 1.25)
        ax_rate.axhline(
            overall_rate,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Overall ({overall_rate:.2%})",
        )
        for i, rate in enumerate(target_rates):
            ax_rate.text(
                i,
                rate,
                f"{rate:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#2f6f6d",
                fontweight="bold",
            )
        ax_rate.legend(loc="upper right")

    ax_rate.set_xlabel(bucket_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)

    train = pd.read_csv("train_clean.csv")
    test = pd.read_csv("test_clean.csv")
    train, test = add_fe(train, test)

    target_col = "TARGET"
    overall_rate = train[target_col].mean()
    print(f"Overall unsatisfied rate: {overall_rate:.4%}")
    print(f"Total samples: {len(train)}")
    print()

    all_stats = []

    # ---- 1. var15 (age) bucketing ----
    bins = [-np.inf, 23, 30, 45, 60, np.inf]
    labels = ["<23", "23-30", "30-45", "45-60", "60+"]
    train["var15_bucket"] = pd.cut(
        train["var15"], bins=bins, labels=labels, right=False
    )
    var15_stats = bucket_stats(train, "var15_bucket", target_col)
    print("=== var15 (age) ===")
    print(var15_stats.to_string(index=False))
    print()
    plot_bucket(
        var15_stats,
        "var15_bucket",
        "var15 (age) vs Unsatisfied rate",
        overall_rate,
        "business_var15_age.png",
        show_sample_size=True,
    )
    var15_stats["feature"] = "var15"
    var15_stats = var15_stats.rename(columns={"var15_bucket": "bucket"})
    all_stats.append(var15_stats)

    # ---- 2a. saldo_var30 binary (0 vs >0) ----
    train["saldo_var30_binary"] = np.where(train["saldo_var30"] > 0, ">0", "=0")
    saldo_bin_stats = bucket_stats(train, "saldo_var30_binary", target_col)
    saldo_bin_stats = saldo_bin_stats.set_index("saldo_var30_binary").loc[
        ["=0", ">0"]
    ].reset_index()
    print("=== saldo_var30 binary ===")
    print(saldo_bin_stats.to_string(index=False))
    print()
    plot_bucket(
        saldo_bin_stats,
        "saldo_var30_binary",
        "saldo_var30 (=0 vs >0) vs Unsatisfied rate",
        overall_rate,
        "business_saldo_var30_binary.png",
        show_sample_size=True,
    )
    saldo_bin_stats["feature"] = "saldo_var30_binary"
    saldo_bin_stats = saldo_bin_stats.rename(
        columns={"saldo_var30_binary": "bucket"}
    )
    all_stats.append(saldo_bin_stats)

    # ---- 2b. saldo_var30 quintile ----
    # qcut with duplicates="drop" because many zeros may cause duplicate edges
    try:
        train["saldo_var30_quintile"] = pd.qcut(
            train["saldo_var30"], q=5, duplicates="drop"
        )
        saldo_q_stats = bucket_stats(train, "saldo_var30_quintile", target_col)
        # convert interval to readable string
        saldo_q_stats["saldo_var30_quintile"] = saldo_q_stats[
            "saldo_var30_quintile"
        ].astype(str)
        print("=== saldo_var30 quintile ===")
        print(saldo_q_stats.to_string(index=False))
        print()
        plot_bucket(
            saldo_q_stats,
            "saldo_var30_quintile",
            "saldo_var30 (quintile) vs Unsatisfied rate",
            overall_rate,
            "business_saldo_var30_quintile.png",
            show_sample_size=True,
        )
        saldo_q_stats["feature"] = "saldo_var30_quintile"
        saldo_q_stats = saldo_q_stats.rename(
            columns={"saldo_var30_quintile": "bucket"}
        )
        all_stats.append(saldo_q_stats)
    except ValueError as e:
        print(f"[warn] saldo_var30 quintile failed: {e}")

    # ---- 3. ind_var30 (0/1) ----
    ind30_stats = bucket_stats(train, "ind_var30", target_col)
    print("=== ind_var30 ===")
    print(ind30_stats.to_string(index=False))
    print()
    plot_bucket(
        ind30_stats,
        "ind_var30",
        "ind_var30 (0/1) vs Unsatisfied rate",
        overall_rate,
        "business_ind_var30.png",
        show_sample_size=True,
    )
    ind30_stats["feature"] = "ind_var30"
    ind30_stats = ind30_stats.rename(columns={"ind_var30": "bucket"})
    all_stats.append(ind30_stats)

    # ---- 4. ind_var26_cte (0/1) ----
    ind26_stats = bucket_stats(train, "ind_var26_cte", target_col)
    print("=== ind_var26_cte ===")
    print(ind26_stats.to_string(index=False))
    print()
    plot_bucket(
        ind26_stats,
        "ind_var26_cte",
        "ind_var26_cte (0/1) vs Unsatisfied rate",
        overall_rate,
        "business_ind_var26_cte.png",
        show_sample_size=True,
    )
    ind26_stats["feature"] = "ind_var26_cte"
    ind26_stats = ind26_stats.rename(columns={"ind_var26_cte": "bucket"})
    all_stats.append(ind26_stats)

    # ---- combined CSV ----
    combined = pd.concat(all_stats, ignore_index=True)
    combined = combined[
        ["feature", "bucket", "sample_size", "n_unsatisfied", "target_rate", "lift"]
    ]
    combined.to_csv("business_top_features_summary.csv", index=False)

    print("Saved: business_var15_age.png")
    print("Saved: business_saldo_var30_binary.png")
    print("Saved: business_saldo_var30_quintile.png")
    print("Saved: business_ind_var30.png")
    print("Saved: business_ind_var26_cte.png")
    print("Saved: business_top_features_summary.csv")


if __name__ == "__main__":
    main()
