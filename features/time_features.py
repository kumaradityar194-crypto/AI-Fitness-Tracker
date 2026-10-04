from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_DIR = PROJECT_ROOT / "data" / "features"


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def create_time_features(
    df,
    signal_columns
):
    """
    Create time-domain features for every window.

    For each signal:
        - mean
        - standard deviation

    One row in the output represents one window.
    """

    # --------------------------------------------------------
    # Group all samples belonging to the same window
    # --------------------------------------------------------

    grouped = df.groupby(
        "window_id",
        sort=False
    )

    feature_data = []

    # --------------------------------------------------------
    # Process each window
    # --------------------------------------------------------

    for window_id, window in grouped:

        features = {}

        # ----------------------------------------------------
        # Window ID
        # ----------------------------------------------------

        features["window_id"] = window_id

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_columns = [
            "participant",
            "exercise",
            "intensity",
            "source_file",
            "window_start_s",
            "window_end_s"
        ]

        for column in metadata_columns:

            if column in window.columns:

                features[column] = window[
                    column
                ].iloc[0]

        # ----------------------------------------------------
        # Mean + Standard Deviation
        # ----------------------------------------------------

        for signal in signal_columns:

            if signal not in window.columns:
                continue

            values = pd.to_numeric(
                window[signal],
                errors="coerce"
            ).dropna()

            if len(values) == 0:
                continue

            # Mean
            features[
                f"{signal}_mean"
            ] = values.mean()

            # Standard deviation
            features[
                f"{signal}_std"
            ] = values.std()

        feature_data.append(
            features
        )

    # --------------------------------------------------------
    # Convert list to DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame(
        feature_data
    )

    return result


# ============================================================
# 3. ACCELEROMETER TIME FEATURES
# ============================================================

def process_accelerometer():

    input_file = (
        FEATURES_DIR /
        "magnitude_accelerometer.csv"
    )

    output_file = (
        FEATURES_DIR /
        "time_features_accelerometer.csv"
    )

    print("\n" + "=" * 60)
    print("ACCELEROMETER TIME FEATURES")
    print("=" * 60)

    if not input_file.exists():

        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_csv(
        input_file
    )

    print(
        f"\nInput shape: {df.shape}"
    )

    if "window_id" not in df.columns:

        raise ValueError(
            "window_id column not found."
        )

    # --------------------------------------------------------
    # Signals used for time-domain features
    # --------------------------------------------------------

    signal_columns = [
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
        "acc_magnitude_filtered"
    ]

    result = create_time_features(
        df=df,
        signal_columns=signal_columns
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nNumber of windows: "
        f"{len(result)}"
    )

    print(
        f"Output shape: "
        f"{result.shape}"
    )

    print("\nCreated features:")

    for column in result.columns:

        print(
            f"  - {column}"
        )

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# 4. GYROSCOPE TIME FEATURES
# ============================================================

def process_gyroscope():

    input_file = (
        FEATURES_DIR /
        "magnitude_gyroscope.csv"
    )

    output_file = (
        FEATURES_DIR /
        "time_features_gyroscope.csv"
    )

    print("\n" + "=" * 60)
    print("GYROSCOPE TIME FEATURES")
    print("=" * 60)

    if not input_file.exists():

        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_csv(
        input_file
    )

    print(
        f"\nInput shape: {df.shape}"
    )

    if "window_id" not in df.columns:

        raise ValueError(
            "window_id column not found."
        )

    # --------------------------------------------------------
    # Signals used for time-domain features
    # --------------------------------------------------------

    signal_columns = [
        "gx_filtered",
        "gy_filtered",
        "gz_filtered",
        "gyro_magnitude_filtered"
    ]

    result = create_time_features(
        df=df,
        signal_columns=signal_columns
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nNumber of windows: "
        f"{len(result)}"
    )

    print(
        f"Output shape: "
        f"{result.shape}"
    )

    print("\nCreated features:")

    for column in result.columns:

        print(
            f"  - {column}"
        )

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - TIME FEATURES")
    print("=" * 60)

    process_accelerometer()

    process_gyroscope()

    print("\n" + "=" * 60)
    print("TIME FEATURES COMPLETED ✅")
    print("=" * 60)


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":
    main()
    
    