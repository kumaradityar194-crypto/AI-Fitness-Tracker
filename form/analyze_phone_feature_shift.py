from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_feature_dataset.csv"
)

PHONE_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_final_feature_dataset.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "form_anomaly_metadata.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_feature_shift.csv"
)


def load_feature_columns():
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    feature_columns = metadata.get(
        "feature_columns"
    )

    if not feature_columns:
        raise ValueError(
            "Feature columns not found in metadata."
        )

    return feature_columns


def prepare_dataframe(
    df,
    feature_columns,
):
    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    for column in feature_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=feature_columns
    )

    return df


def main():
    print()
    print("=" * 80)
    print("AI FITNESS TRACKER - PHONE FEATURE SHIFT ANALYSIS")
    print("=" * 80)

    feature_columns = load_feature_columns()

    print()
    print(
        f"Total features: {len(feature_columns)}"
    )

    train_df = pd.read_csv(
        TRAIN_FEATURE_FILE
    )

    phone_df = pd.read_csv(
        PHONE_FEATURE_FILE
    )

    train_squat_df = train_df[
        train_df["exercise"]
        .astype(str)
        .str.lower()
        .str.strip()
        == "squat"
    ].copy()

    train_squat_df = prepare_dataframe(
        train_squat_df,
        feature_columns,
    )

    phone_df = prepare_dataframe(
        phone_df,
        feature_columns,
    )

    print()
    print(
        f"Training squat rows: {len(train_squat_df)}"
    )

    print(
        f"Phone rows: {len(phone_df)}"
    )

    rows = []

    for feature in feature_columns:
        train_values = (
            train_squat_df[feature]
            .to_numpy(
                dtype=float
            )
        )

        phone_values = (
            phone_df[feature]
            .to_numpy(
                dtype=float
            )
        )

        train_mean = float(
            np.mean(train_values)
        )

        train_std = float(
            np.std(train_values)
        )

        phone_mean = float(
            np.mean(phone_values)
        )

        phone_std = float(
            np.std(phone_values)
        )

        safe_std = max(
            train_std,
            1e-8,
        )

        mean_difference = (
            phone_mean
            - train_mean
        )

        standardized_shift = (
            mean_difference
            / safe_std
        )

        rows.append(
            {
                "feature": feature,
                "training_mean": train_mean,
                "training_std": train_std,
                "phone_mean": phone_mean,
                "phone_std": phone_std,
                "mean_difference": mean_difference,
                "standardized_shift": standardized_shift,
                "absolute_standardized_shift": abs(
                    standardized_shift
                ),
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df = result_df.sort_values(
        "absolute_standardized_shift",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result_df) + 1,
        ),
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("TOP 15 FEATURE SHIFTS")
    print("=" * 80)

    display_columns = [
        "rank",
        "feature",
        "training_mean",
        "phone_mean",
        "standardized_shift",
        "absolute_standardized_shift",
    ]

    print(
        result_df[
            display_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print("=" * 80)
    print("PHONE FEATURE SHIFT ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()