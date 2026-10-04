from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


ACC_INPUT = (
    PROCESSED_DIR /
    "phone_accelerometer_magnitude.csv"
)

GYRO_INPUT = (
    PROCESSED_DIR /
    "phone_gyroscope_magnitude.csv"
)

ACC_OUTPUT = (
    PROCESSED_DIR /
    "phone_time_features_accelerometer.csv"
)

GYRO_OUTPUT = (
    PROCESSED_DIR /
    "phone_time_features_gyroscope.csv"
)


# ============================================================
# 2. TIME FEATURE FUNCTION
# ============================================================

def create_time_features(
    df,
    signal_columns
):
    """
    Create mean and standard deviation
    for every window.
    """

    grouped = df.groupby(
        "window_id",
        sort=False
    )

    feature_rows = []

    for window_id, window in grouped:

        features = {}

        # Window ID
        features["window_id"] = window_id

        # Calculate features
        for signal in signal_columns:

            if signal not in window.columns:
                continue

            values = pd.to_numeric(
                window[signal],
                errors="coerce"
            ).dropna()

            if len(values) == 0:
                continue

            features[
                f"{signal}_mean"
            ] = values.mean()

            features[
                f"{signal}_std"
            ] = values.std()

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


# ============================================================
# 3. ACCELEROMETER TIME FEATURES
# ============================================================

def process_accelerometer():

    print("\n" + "=" * 60)
    print("PHONE ACCELEROMETER TIME FEATURES")
    print("=" * 60)

    df = pd.read_csv(ACC_INPUT)

    print("Input shape:", df.shape)

    signal_columns = [
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
        "acc_magnitude_filtered"
    ]

    missing = [
        col
        for col in signal_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing accelerometer columns: {missing}"
        )

    result = create_time_features(
        df=df,
        signal_columns=signal_columns
    )

    result.to_csv(
        ACC_OUTPUT,
        index=False
    )

    print(
        "Number of windows:",
        len(result)
    )

    print(
        "Output shape:",
        result.shape
    )

    print("\nCreated features:")

    for col in result.columns:
        print(" ", col)

    print("\nSaved:")
    print(ACC_OUTPUT)


# ============================================================
# 4. GYROSCOPE TIME FEATURES
# ============================================================

def process_gyroscope():

    print("\n" + "=" * 60)
    print("PHONE GYROSCOPE TIME FEATURES")
    print("=" * 60)

    df = pd.read_csv(GYRO_INPUT)

    print("Input shape:", df.shape)

    signal_columns = [
        "gx_filtered",
        "gy_filtered",
        "gz_filtered",
        "gyro_magnitude_filtered"
    ]

    missing = [
        col
        for col in signal_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing gyroscope columns: {missing}"
        )

    result = create_time_features(
        df=df,
        signal_columns=signal_columns
    )

    result.to_csv(
        GYRO_OUTPUT,
        index=False
    )

    print(
        "Number of windows:",
        len(result)
    )

    print(
        "Output shape:",
        result.shape
    )

    print("\nCreated features:")

    for col in result.columns:
        print(" ", col)

    print("\nSaved:")
    print(GYRO_OUTPUT)


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - PHONE TIME FEATURES")
    print("=" * 60)

    process_accelerometer()

    process_gyroscope()

    print("\n" + "=" * 60)
    print("PHONE TIME FEATURES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()