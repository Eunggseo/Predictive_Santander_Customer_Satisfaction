from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("xgb_5cv_5seed_outputs")


def read_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["ID", "TARGET"]].copy()


def weighted_average(name: str, weighted_paths: list[tuple[str, float]]) -> None:
    base = read_submission(weighted_paths[0][0])
    pred = base["TARGET"] * weighted_paths[0][1]
    total_weight = weighted_paths[0][1]

    for path, weight in weighted_paths[1:]:
        df = read_submission(path)
        if not base["ID"].equals(df["ID"]):
            raise ValueError(f"ID order mismatch: {path}")
        pred += df["TARGET"] * weight
        total_weight += weight

    output = pd.DataFrame({"ID": base["ID"], "TARGET": pred / total_weight})
    output_path = OUTPUT_DIR / name
    output.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main() -> None:
    xgb_5seed = str(OUTPUT_DIR / "submission_xgb_5cv_5seed.csv")

    weighted_average(
        "submission_xgb5_lgbm12_simple_average.csv",
        [
            (xgb_5seed, 1.0),
            ("submission_lightgbm_fe_spw_12.csv", 1.0),
        ],
    )
    weighted_average(
        "submission_xgb5_strong3_p2_simple_average.csv",
        [
            (xgb_5seed, 1.0),
            ("submission_strong3_power_p2.csv", 1.0),
        ],
    )
    weighted_average(
        "submission_xgb5_strong3_p2_xgb30.csv",
        [
            (xgb_5seed, 0.3),
            ("submission_strong3_power_p2.csv", 0.7),
        ],
    )


if __name__ == "__main__":
    main()
