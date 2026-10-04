from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_DIR = PROJECT_ROOT / "data" / "features"


# ============================================================
# 2. RECORDING KEY
# ============================================================

def create_recording_key(source_file):
    """
    Create the same recording key for corresponding
    accelerometer and gyroscope files.
    """

    if pd.isna(source_file):

        return None

    key = Path(
        str(source_file)
    ).stem

    key = key.replace(
        "_Accelerometer_12.500Hz",
        ""
    )

    key = key.replace(
        "_Gyroscope_25.000Hz",
        ""
    )

    return key


# ============================================================
# 3. LOCAL WINDOW NUMBER
# ============================================================

def add_recording_window_number(df):

    df = df.copy()

    df = df.sort_values(
        [
            "source_file",
            "window_start_s"
        ]
    )

    df["recording_window_number"] = (
        df.groupby(
            "source_file",
            sort=False
        ).cumcount()
    )

    return df


# ============================================================
# 4. LOAD TIME + FREQUENCY FEATURES
# ============================================================

def load_sensor_features(sensor):

    time_file = (
        FEATURES_DIR /
        f"time_features_{sensor}.csv"
    )

    frequency_file = (
        FEATURES_DIR /
        f"frequency_features_{sensor}.csv"
    )

    if not time_file.exists():

        raise FileNotFoundError(
            f"Missing:\n{time_file}"
        )

    if not frequency_file.exists():

        raise FileNotFoundError(
            f"Missing:\n{frequency_file}"
        )

    print(
        f"\nReading:\n{time_file.name}"
    )

    time_df = pd.read_csv(
        time_file
    )

    print(
        f"Reading:\n{frequency_file.name}"
    )

    frequency_df = pd.read_csv(
        frequency_file
    )

    merge_keys = [
        "window_id",
        "source_file"
    ]

    # --------------------------------------------------------
    # Merge time + frequency
    # --------------------------------------------------------

    df = pd.merge(
        time_df,
        frequency_df,
        on=merge_keys,
        how="inner",
        suffixes=(
            "",
            "_freq"
        )
    )

    # --------------------------------------------------------
    # Metadata columns
    # --------------------------------------------------------

    metadata_columns = [
        "participant",
        "exercise",
        "intensity",
        "window_start_s",
        "window_end_s"
    ]

    # --------------------------------------------------------
    # Keep only one copy of metadata
    # --------------------------------------------------------

    for column in metadata_columns:

        frequency_column = f"{column}_freq"

        if frequency_column in df.columns:

            if column not in df.columns:

                df[column] = df[
                    frequency_column
                ]

            df = df.drop(
                columns=[
                    frequency_column
                ]
            )

    # --------------------------------------------------------
    # Create recording key
    # --------------------------------------------------------

    df["recording_key"] = (
        df["source_file"]
        .apply(
            create_recording_key
        )
    )

    # --------------------------------------------------------
    # Create local window number
    # --------------------------------------------------------

    df = add_recording_window_number(
        df
    )

    return df


# ============================================================
# 5. ADD RAW MAGNITUDE FEATURES
# ============================================================

def add_magnitude_features(
    feature_df,
    sensor
):

    magnitude_file = (
        FEATURES_DIR /
        f"magnitude_{sensor}.csv"
    )

    if not magnitude_file.exists():

        raise FileNotFoundError(
            f"Missing:\n{magnitude_file}"
        )

    magnitude_df = pd.read_csv(
        magnitude_file
    )

    if sensor == "accelerometer":

        magnitude_column = (
            "acc_magnitude"
        )

        prefix = "acc"

    elif sensor == "gyroscope":

        magnitude_column = (
            "gyro_magnitude"
        )

        prefix = "gyro"

    else:

        raise ValueError(
            "Invalid sensor."
        )

    # --------------------------------------------------------
    # Aggregate raw magnitude
    # --------------------------------------------------------

    magnitude_features = (
        magnitude_df
        .groupby(
            [
                "window_id",
                "source_file"
            ],
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

    # --------------------------------------------------------
    # Rename
    # --------------------------------------------------------

    magnitude_features = (
        magnitude_features.rename(
            columns={
                "mean":
                    f"{prefix}_magnitude_mean",

                "std":
                    f"{prefix}_magnitude_std",

                "min":
                    f"{prefix}_magnitude_min",

                "max":
                    f"{prefix}_magnitude_max"
            }
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    feature_df = pd.merge(
        feature_df,
        magnitude_features,
        on=[
            "window_id",
            "source_file"
        ],
        how="left"
    )

    return feature_df


# ============================================================
# 6. PREFIX SENSOR FEATURE COLUMNS
# ============================================================

def prefix_sensor_features(
    df,
    prefix
):

    df = df.copy()

    protected_columns = [
        "window_id",
        "source_file",
        "recording_key",
        "recording_window_number",
        "participant",
        "exercise",
        "intensity",
        "window_start_s",
        "window_end_s"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in protected_columns
    ]

    df = df.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in feature_columns
        }
    )

    return df


# ============================================================
# 7. PREPARE ACCELEROMETER
# ============================================================

def prepare_accelerometer(df):

    df = prefix_sensor_features(
        df,
        "acc"
    )

    return df


# ============================================================
# 8. PREPARE GYROSCOPE
# ============================================================

def prepare_gyroscope(df):

    df = prefix_sensor_features(
        df,
        "gyro"
    )

    return df


# ============================================================
# 9. ALIGN ACCELEROMETER + GYROSCOPE
# ============================================================

def align_sensors(
    accelerometer_df,
    gyroscope_df
):

    acc = accelerometer_df.copy()
    gyro = gyroscope_df.copy()

    # --------------------------------------------------------
    # Alignment keys
    # --------------------------------------------------------

    alignment_keys = [
        "recording_key",
        "recording_window_number"
    ]

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_acc = alignment_keys + [
        "participant",
        "exercise",
        "intensity"
    ]

    required_gyro = alignment_keys

    for column in required_acc:

        if column not in acc.columns:

            raise ValueError(
                f"Accelerometer missing: "
                f"{column}"
            )

    for column in required_gyro:

        if column not in gyro.columns:

            raise ValueError(
                f"Gyroscope missing: "
                f"{column}"
            )

    # --------------------------------------------------------
    # Remove identifiers that should NOT enter ML features
    # --------------------------------------------------------

    gyro = gyro.drop(
        columns=[
            "window_id",
            "source_file",
            "participant",
            "exercise",
            "intensity",
            "window_start_s",
            "window_end_s"
        ],
        errors="ignore"
    )

    acc = acc.drop(
        columns=[
            "window_id",
            "source_file"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Check duplicate keys
    # --------------------------------------------------------

    acc_duplicates = (
        acc.duplicated(
            subset=alignment_keys
        ).sum()
    )

    gyro_duplicates = (
        gyro.duplicated(
            subset=alignment_keys
        ).sum()
    )

    print(
        f"\nDuplicate accelerometer keys: "
        f"{acc_duplicates}"
    )

    print(
        f"Duplicate gyroscope keys: "
        f"{gyro_duplicates}"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    acc = acc.drop_duplicates(
        subset=alignment_keys,
        keep="first"
    )

    gyro = gyro.drop_duplicates(
        subset=alignment_keys,
        keep="first"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    combined = pd.merge(
        acc,
        gyro,
        on=alignment_keys,
        how="inner"
    )

    if combined.empty:

        raise ValueError(
            "No accelerometer and gyroscope "
            "windows could be aligned."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    combined = combined.sort_values(
        alignment_keys
    )

    combined = combined.reset_index(
        drop=True
    )

    return combined


# ============================================================
# 10. FINAL DATASET PREPARATION
# ============================================================

def prepare_final_dataset(df):

    df = df.copy()

    # --------------------------------------------------------
    # Remove ALL internal / non-feature columns
    # --------------------------------------------------------

    columns_to_drop = [
        "window_id",
        "source_file",
        "window_start_s",
        "window_end_s",
        "recording_key"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # --------------------------------------------------------
    # recording_window_number is useful internally,
    # but should NOT be an ML feature.
    # --------------------------------------------------------

    df = df.drop(
        columns=[
            "recording_window_number"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    after = len(df)

    print(
        f"\nDuplicate final rows removed: "
        f"{before - after}"
    )

    # --------------------------------------------------------
    # Important columns first
    # --------------------------------------------------------

    first_columns = [
        "participant",
        "exercise",
        "intensity"
    ]

    existing_first = [
        column
        for column in first_columns
        if column in df.columns
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in existing_first
    ]

    df = df[
        existing_first +
        remaining_columns
    ]

    return df


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

def print_summary(df):

    print("\n" + "=" * 60)
    print("FINAL FEATURE DATASET SUMMARY")
    print("=" * 60)

    print(
        f"\nDataset shape: {df.shape}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    # --------------------------------------------------------
    # Number of ML features
    # --------------------------------------------------------

    metadata_columns = [
        "participant",
        "exercise",
        "intensity"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    print(
        f"ML feature count: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Exercise distribution
    # --------------------------------------------------------

    print(
        "\nExercise distribution:"
    )

    print(
        df["exercise"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Participant distribution
    # --------------------------------------------------------

    print(
        "\nParticipant distribution:"
    )

    print(
        df["participant"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Intensity distribution
    # --------------------------------------------------------

    print(
        "\nIntensity distribution:"
    )

    print(
        df["intensity"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print(
        "\nMissing values:"
    )

    missing = (
        df
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        print(
            "No missing values ✅"
        )

    else:

        print(
            missing
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    print(
        "\nML feature columns:"
    )

    for column in feature_columns:

        print(
            f"  - {column}"
        )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - FEATURE COMBINATION")
    print("=" * 60)

    # ========================================================
    # ACCELEROMETER
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "LOADING ACCELEROMETER FEATURES"
    )

    print(
        "=" * 60
    )

    acc_df = load_sensor_features(
        "accelerometer"
    )

    print(
        f"\nAccelerometer rows: "
        f"{len(acc_df)}"
    )

    acc_df = add_magnitude_features(
        acc_df,
        "accelerometer"
    )

    acc_df = prepare_accelerometer(
        acc_df
    )

    print(
        f"Accelerometer prepared shape: "
        f"{acc_df.shape}"
    )

    # ========================================================
    # GYROSCOPE
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "LOADING GYROSCOPE FEATURES"
    )

    print(
        "=" * 60
    )

    gyro_df = load_sensor_features(
        "gyroscope"
    )

    print(
        f"\nGyroscope rows: "
        f"{len(gyro_df)}"
    )

    gyro_df = add_magnitude_features(
        gyro_df,
        "gyroscope"
    )

    gyro_df = prepare_gyroscope(
        gyro_df
    )

    print(
        f"Gyroscope prepared shape: "
        f"{gyro_df.shape}"
    )

    # ========================================================
    # ALIGN
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "ALIGNING SENSORS"
    )

    print(
        "=" * 60
    )

    combined_df = align_sensors(
        accelerometer_df=acc_df,
        gyroscope_df=gyro_df
    )

    print(
        f"\nAligned dataset shape: "
        f"{combined_df.shape}"
    )

    # ========================================================
    # FINAL PREPARATION
    # ========================================================

    combined_df = prepare_final_dataset(
        combined_df
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = (
        FEATURES_DIR /
        "final_feature_dataset.csv"
    )

    combined_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        combined_df
    )

    print(
        f"\nFinal dataset saved to:"
    )

    print(
        output_file
    )

    print("\n" + "=" * 60)
    print("FEATURE COMBINATION COMPLETED ✅")
    print("=" * 60)


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":

    main()