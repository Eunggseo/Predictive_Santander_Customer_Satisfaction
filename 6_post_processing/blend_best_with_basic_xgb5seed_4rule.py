from pathlib import Path

import pandas as pd


BEST_PATH = Path("Ensemble output.csv")
XGB5SEED_PATH = Path("basic_xgb_5fold_5seed.csv")
TEST_PATH = Path("test.csv")


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns[:2]) != ["ID", "TARGET"]:
        raise ValueError(f"{path} must start with ID,TARGET columns")
    return df[["ID", "TARGET"]].copy()


def four_rule_low_risk(df: pd.DataFrame) -> pd.Series:
    # ============================================================
    # 4-rule post-processing - FINAL VALIDATED VERSION
    # Re-validated: 2026-05
    # Train union: n=1807, TARGET rate=0.0% (zero-positive)
    # OOF AUC: 0.837639 -> 0.837924 (+0.000285)
    # Early exploration rules (var15<=22, num_aport>4.5, etc.)
    # are superseded by this business-readable version.
    # ============================================================
    return (
        (df["num_aport_var13_hace3"] >= 6)
        | (df["num_meses_var13_largo_ult3"] >= 1)
        | (df["var15"] < 23)
        | (df["var36"] == 0)
    )


def main() -> None:
    best = read_submission(BEST_PATH)
    xgb5seed = read_submission(XGB5SEED_PATH)
    test = pd.read_csv(TEST_PATH)

    if not best["ID"].equals(xgb5seed["ID"]):
        raise ValueError("ID order mismatch between best and xgb5seed")
    if not best["ID"].equals(test["ID"]):
        raise ValueError("ID order mismatch between submission and test")

    rule_mask = four_rule_low_risk(test).values
    print(f"4-rule test rows set to zero: {int(rule_mask.sum())}")

    for xgb_weight in [0.05, 0.10]:
        best_weight = 1.0 - xgb_weight
        target = best_weight * best["TARGET"] + xgb_weight * xgb5seed["TARGET"]
        target.loc[rule_mask] = 0.0

        output_path = Path(
            f"sub_best_basic_xgb5seed_w{xgb_weight:.2f}_4rule.csv".replace(
                "0.", ""
            )
        )
        pd.DataFrame({"ID": best["ID"], "TARGET": target}).to_csv(
            output_path, index=False
        )
        print(
            f"Saved: {output_path} "
            f"({best_weight:.2f} best + {xgb_weight:.2f} xgb5seed, then 4-rule)"
        )


if __name__ == "__main__":
    main()
