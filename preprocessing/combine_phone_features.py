from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"


TIME_ACC_FILE = (
    PROCESSED_DIR /
    "phone_time_features_accelerometer.csv"
)

TIME_GYRO_FILE = (
    PROCESSED_DIR /
    "phone_time_features_gyroscope.csv"
)

FREQ_ACC_FILE = (
    PROCESSED_DIR /
    "phone_frequency_features_accelerometer.csv"
)

FREQ_GYRO_FILE = (
    PROCESSED_DIR /
    "phone_frequency_features_gyroscope.csv"
)

MAG_ACC_FILE = (
    PROCESSED_DIR /
    "phone_accelerometer_magnitude.csv"
)

MAG_GYRO_FILE = (
    PROCESSED_DIR /
    "phone_gyroscope_magnitude.csv"
)

FEATURE_LIST_FILE = (
    MODEL_DIR /
    "exercise_features.pkl"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "phone_final_feature_dataset.csv"
)


# ============================================================
# 2. LOAD
# ============================================================

def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    return pd.read_csv(path)


# ============================================================
# 3. RAW MAGNITUDE FEATURES
# ============================================================

def create_magnitude_features(
    df,
    magnitude_column,
    output_prefix
):
    """
    Create mean, std, min and max of raw magnitude.

    output_prefix is deliberately chosen to match
    the exact feature names expected by the trained model.
    """

    result = (
        df.groupby(
            "window_id",
            sort=False
        )[magnitude_column]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max"
            ]
        )
        .reset_index()
    )

    result = result.rename(
        columns={
            "mean":
                f"{output_prefix}_magnitude_mean",

            "std":
                f"{output_prefix}_magnitude_std",

            "min":
                f"{output_prefix}_magnitude_min",

            "max":
                f"{output_prefix}_magnitude_max"
        }
    )

    return result


# ============================================================
# 4. PREFIX FEATURE COLUMNS
# ============================================================

def prefix_features(df, prefix):

    df = df.copy()

    renamed = {}

    for column in df.columns:

        if column == "window_id":
            continue

        renamed[column] = (
            f"{prefix}_{column}"
        )

    return df.rename(
        columns=renamed
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI FITNESS TRACKER - PHONE FINAL FEATURE DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading feature files...")

    time_acc = load_csv(TIME_ACC_FILE)
    time_gyro = load_csv(TIME_GYRO_FILE)

    freq_acc = load_csv(FREQ_ACC_FILE)
    freq_gyro = load_csv(FREQ_GYRO_FILE)

    mag_acc = load_csv(MAG_ACC_FILE)
    mag_gyro = load_csv(MAG_GYRO_FILE)

    print("Time ACC :", time_acc.shape)
    print("Time GYRO:", time_gyro.shape)
    print("Freq ACC :", freq_acc.shape)
    print("Freq GYRO:", freq_gyro.shape)
    print("Mag ACC  :", mag_acc.shape)
    print("Mag GYRO :", mag_gyro.shape)

    # --------------------------------------------------------
    # Check window IDs
    # --------------------------------------------------------

    counts = {
        "time_acc": time_acc["window_id"].nunique(),
        "time_gyro": time_gyro["window_id"].nunique(),
        "freq_acc": freq_acc["window_id"].nunique(),
        "freq_gyro": freq_gyro["window_id"].nunique(),
        "mag_acc": mag_acc["window_id"].nunique(),
        "mag_gyro": mag_gyro["window_id"].nunique(),
    }

    print("\nWindow counts:")

    for name, count in counts.items():
        print(
            f"  {name}: {count}"
        )

    if len(set(counts.values())) != 1:
        print("\nWindow counts differ across sensors.")
        print("Aligning all feature tables to common window IDs.")

    common_window_ids = set(time_acc["window_id"].unique())
    common_window_ids &= set(time_gyro["window_id"].unique())
    common_window_ids &= set(freq_acc["window_id"].unique())
    common_window_ids &= set(freq_gyro["window_id"].unique())
    common_window_ids &= set(mag_acc["window_id"].unique())
    common_window_ids &= set(mag_gyro["window_id"].unique())

    common_window_ids = sorted(common_window_ids)

    if not common_window_ids:
        raise ValueError(
            "No common window IDs are available across all phone feature files."
        )

    print(
        "\nCommon aligned windows:",
        len(common_window_ids)
    )
    print(
        "Window IDs:",
        common_window_ids
    )

    common_window_ids = set(common_window_ids)

    time_acc = time_acc[
        time_acc["window_id"].isin(common_window_ids)
    ].copy()

    time_gyro = time_gyro[
        time_gyro["window_id"].isin(common_window_ids)
    ].copy()

    freq_acc = freq_acc[
        freq_acc["window_id"].isin(common_window_ids)
    ].copy()

    freq_gyro = freq_gyro[
        freq_gyro["window_id"].isin(common_window_ids)
    ].copy()

    mag_acc = mag_acc[
        mag_acc["window_id"].isin(common_window_ids)
    ].copy()

    mag_gyro = mag_gyro[
        mag_gyro["window_id"].isin(common_window_ids)
    ].copy()

    # --------------------------------------------------------
    # Prefix time features
    # --------------------------------------------------------

    time_acc = prefix_features(
        time_acc,
        "acc"
    )

    time_gyro = prefix_features(
        time_gyro,
        "gyro"
    )

    # --------------------------------------------------------
    # Prefix frequency features
    # --------------------------------------------------------

    freq_acc = prefix_features(
        freq_acc,
        "acc"
    )

    freq_gyro = prefix_features(
        freq_gyro,
        "gyro"
    )

    # --------------------------------------------------------
    # Create EXACT model-compatible magnitude names
    # --------------------------------------------------------

    acc_mag_features = (
        create_magnitude_features(
            mag_acc,
            "acc_magnitude",
            "acc_acc"
        )
    )

    gyro_mag_features = (
        create_magnitude_features(
            mag_gyro,
            "gyro_magnitude",
            "gyro_gyro"
        )
    )

    # --------------------------------------------------------
    # Merge all features
    # --------------------------------------------------------

    print("\nCombining features...")

    final_df = time_acc.merge(
        freq_acc,
        on="window_id",
        how="inner"
    )

    final_df = final_df.merge(
        time_gyro,
        on="window_id",
        how="inner"
    )

    final_df = final_df.merge(
        freq_gyro,
        on="window_id",
        how="inner"
    )

    final_df = final_df.merge(
        acc_mag_features,
        on="window_id",
        how="inner"
    )

    final_df = final_df.merge(
        gyro_mag_features,
        on="window_id",
        how="inner"
    )

    print(
        "Combined shape:",
        final_df.shape
    )

    # --------------------------------------------------------
    # Load exact feature list used during training
    # --------------------------------------------------------

    expected_features = joblib.load(
        FEATURE_LIST_FILE
    )

    print(
        "\nModel expects:",
        len(expected_features),
        "features"
    )

    # --------------------------------------------------------
    # Missing / extra features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in expected_features
        if feature not in final_df.columns
    ]

    extra_features = [
        column
        for column in final_df.columns
        if column != "window_id"
        and column not in expected_features
    ]

    if missing_features:

        print("\nMISSING FEATURES:")

        for feature in missing_features:
            print(
                "  ",
                feature
            )

        raise ValueError(
            "Required model features are missing."
        )

    if extra_features:

        print("\nEXTRA FEATURES:")

        for feature in extra_features:
            print(
                "  ",
                feature
            )

    # --------------------------------------------------------
    # EXACT FEATURE ORDER
    # --------------------------------------------------------

    feature_df = final_df[
        ["window_id"] +
        list(expected_features)
    ].copy()

    # --------------------------------------------------------
    # Verify feature count
    # --------------------------------------------------------

    feature_count = (
        feature_df.shape[1] - 1
    )

    print(
        "\nFinal ML feature count:",
        feature_count
    )

    if feature_count != 48:

        raise ValueError(
            f"Expected 48 features, "
            f"got {feature_count}."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = (
        feature_df[
            expected_features
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Total missing feature values:",
        missing_values
    )

    if missing_values > 0:

        raise ValueError(
            "Final dataset contains missing values."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL PHONE FEATURE DATASET")
    print("=" * 70)

    print(
        "\nDataset shape:",
        feature_df.shape
    )

    print(
        "Windows:",
        feature_df["window_id"].nunique()
    )

    print(
        "ML features:",
        feature_count
    )

    print("\nFeature names:")

    for feature in expected_features:
        print(
            "  ",
            feature
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 70)
    print("PHONE 48-FEATURE DATASET READY")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()