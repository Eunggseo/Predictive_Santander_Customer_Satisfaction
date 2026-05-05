"""
Final Santander submission pipeline.

Reproduces the final Private AUC 0.82642 submission:
1. Austin stacked ensemble submission (Private 0.82606)
2. XGBoost FE + validated 4-rule submission
3. Rank-average both prediction files
4. Apply the final validated 4-rule low-risk post-processing

Output:
- results/submissions/sub4_rankavg_4rule.csv
- sub4_rankavg_4rule.csv, kept for compatibility with the submitted filename
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "submissions"

AUSTIN_STACKED_PATH = RESULTS_DIR / "austin_stacked_ensemble_082606.csv"
XGB_FE_4RULE_PATH = RESULTS_DIR / "xgboost_fe_4rule_postprocess.csv"
TEST_PATH = ROOT / "test.csv"
FINAL_OUTPUT_PATH = RESULTS_DIR / "sub4_rankavg_4rule.csv"
ROOT_COMPAT_OUTPUT_PATH = ROOT / "sub4_rankavg_4rule.csv"


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns[:2]) != ["ID", "TARGET"]:
        raise ValueError(f"{path} must start with ID,TARGET columns")
    return df[["ID", "TARGET"]].copy()


def rank_average(a: pd.Series, b: pd.Series) -> pd.Series:
    rank_a = a.rank(method="average") / len(a)
    rank_b = b.rank(method="average") / len(b)
    return (rank_a + rank_b) / 2


def final_4rule_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["num_aport_var13_hace3"] >= 6)
        | (df["num_meses_var13_largo_ult3"] >= 1)
        | (df["var15"] < 23)
        | (df["var36"] == 0)
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stacked = read_submission(AUSTIN_STACKED_PATH)
    xgb_fe = read_submission(XGB_FE_4RULE_PATH)
    test = pd.read_csv(TEST_PATH)

    if not stacked["ID"].equals(xgb_fe["ID"]):
        raise ValueError("ID order mismatch between stacked and XGBoost submissions")
    if not stacked["ID"].equals(test["ID"]):
        raise ValueError("ID order mismatch between submissions and test.csv")

    target = rank_average(stacked["TARGET"], xgb_fe["TARGET"])
    mask_4rule = final_4rule_mask(test)
    target.loc[mask_4rule.values] = 0.0

    output = pd.DataFrame({"ID": stacked["ID"], "TARGET": target})
    output.to_csv(FINAL_OUTPUT_PATH, index=False)
    output.to_csv(ROOT_COMPAT_OUTPUT_PATH, index=False)

    print(f"4-rule test rows set to zero: {int(mask_4rule.sum())}")
    print(f"Saved: {FINAL_OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Saved: {ROOT_COMPAT_OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
