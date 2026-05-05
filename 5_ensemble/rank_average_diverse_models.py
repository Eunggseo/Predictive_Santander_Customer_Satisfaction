import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = "xgb_5cv_5seed_outputs"


def read_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns[:2]) != ["ID", "TARGET"]:
        raise ValueError(f"{path} must start with ID,TARGET columns")
    return df[["ID", "TARGET"]].copy()


def safe_stem(path: str) -> str:
    return Path(path).stem.replace("submission_", "")


def rank_average(submissions: list[pd.DataFrame], weights: list[float] | None) -> pd.Series:
    ranks = [df["TARGET"].rank(method="average") / len(df) for df in submissions]
    rank_df = pd.concat(ranks, axis=1)
    if weights is None:
        return rank_df.mean(axis=1)
    return rank_df.mul(weights, axis=1).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a rank-average blend from multiple submission files."
    )
    parser.add_argument(
        "submissions",
        nargs="+",
        help="Submission CSV files with ID,TARGET columns. Use at least two files.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", help="Optional output name without .csv")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Optional rank weights, one per submission. They will be normalized.",
    )
    args = parser.parse_args()

    if len(args.submissions) < 2:
        raise ValueError("Provide at least two submission files")
    if args.weights and len(args.weights) != len(args.submissions):
        raise ValueError("Number of weights must match number of submissions")
    weights = None
    if args.weights:
        weight_sum = sum(args.weights)
        if weight_sum <= 0:
            raise ValueError("Weights must sum to a positive value")
        weights = [weight / weight_sum for weight in args.weights]

    dfs = [read_submission(path) for path in args.submissions]
    base_id = dfs[0]["ID"]
    for path, df in zip(args.submissions[1:], dfs[1:]):
        if not base_id.equals(df["ID"]):
            raise ValueError(f"ID order mismatch for {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    name = args.name
    if not name:
        label = "_".join(safe_stem(path) for path in args.submissions)
        suffix = "weighted_rank_average" if weights else "rank_average"
        name = f"submission_{label}_{suffix}"

    output_path = output_dir / f"{name}.csv"
    pd.DataFrame({"ID": base_id, "TARGET": rank_average(dfs, weights)}).to_csv(
        output_path, index=False
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
