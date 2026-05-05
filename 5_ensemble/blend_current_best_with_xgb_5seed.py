import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CURRENT_BEST = "submission_lightgbm_fe_spw_12.csv"
DEFAULT_XGB_5SEED = "xgb_5cv_5seed_outputs/submission_xgb_5cv_5seed.csv"
DEFAULT_OUTPUT_DIR = "xgb_5cv_5seed_outputs"


def read_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns[:2]) != ["ID", "TARGET"]:
        raise ValueError(f"{path} must start with ID,TARGET columns")
    return df[["ID", "TARGET"]].copy()


def minmax_rank_average(a: pd.Series, b: pd.Series) -> pd.Series:
    rank_a = a.rank(method="average") / len(a)
    rank_b = b.rank(method="average") / len(b)
    return (rank_a + rank_b) / 2


def safe_stem(path: str) -> str:
    return Path(path).stem.replace("submission_", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blend the current best submission with xgb_5cv_5seed."
    )
    parser.add_argument("--current-best", default=DEFAULT_CURRENT_BEST)
    parser.add_argument("--xgb-5seed", default=DEFAULT_XGB_5SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    current = read_submission(args.current_best)
    xgb = read_submission(args.xgb_5seed)

    if not current["ID"].equals(xgb["ID"]):
        raise ValueError("ID order mismatch between current_best and xgb_5seed")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    label = safe_stem(args.current_best)

    blends = {
        f"submission_{label}_xgb5_w50.csv": (
            0.5 * current["TARGET"] + 0.5 * xgb["TARGET"]
        ),
        f"submission_{label}_xgb5_w30.csv": (
            0.7 * current["TARGET"] + 0.3 * xgb["TARGET"]
        ),
        f"submission_{label}_xgb5_rank_average.csv": minmax_rank_average(
            current["TARGET"], xgb["TARGET"]
        ),
    }

    for filename, target in blends.items():
        output = pd.DataFrame({"ID": current["ID"], "TARGET": target})
        output_path = output_dir / filename
        output.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
