from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

FEATURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FFT FEATURE FUNCTION
# ============================================================

def calculate_frequency_features(
    values,
    sampling_rate
):
    """
    Calculate three frequency-domain features:

    1. Dominant frequency
    2. Weighted average frequency
    3. Normalized spectral entropy
    """

    values = np.asarray(
        values,
        dtype=float
    )

    # Remove invalid values
    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan, np.nan, np.nan

    # --------------------------------------------------------
    # Remove DC component
    # --------------------------------------------------------

    values = (
        values -
        np.mean(values)
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    fft_values = np.fft.rfft(
        values
    )

    frequencies = np.fft.rfftfreq(
        len(values),
        d=1 / sampling_rate
    )

    # Power spectrum
    power = (
        np.abs(fft_values) ** 2
    )

    # --------------------------------------------------------
    # Remove zero-frequency component
    # --------------------------------------------------------

    if len(power) > 1:

        frequencies = frequencies[1:]
        power = power[1:]

    # --------------------------------------------------------
    # No usable power
    # --------------------------------------------------------

    total_power = np.sum(power)

    if total_power <= 0:

        return (
            0.0,
            0.0,
            0.0
        )

    # ========================================================
    # 1. DOMINANT FREQUENCY
    # ========================================================

    dominant_index = np.argmax(
        power
    )

    dominant_frequency = (
        frequencies[
            dominant_index
        ]
    )

    # ========================================================
    # 2. WEIGHTED AVERAGE FREQUENCY
    # ========================================================

    weighted_frequency = (
        np.sum(
            frequencies * power
        )
        /
        total_power
    )

    # ========================================================
    # 3. SPECTRAL ENTROPY
    # ========================================================

    probability = (
        power /
        total_power
    )

    probability = probability[
        probability > 0
    ]

    spectral_entropy = -np.sum(
        probability *
        np.log2(probability)
    )

    # --------------------------------------------------------
    # Normalize entropy between 0 and 1
    # --------------------------------------------------------

    if len(probability) > 1:

        spectral_entropy = (
            spectral_entropy /
            np.log2(len(probability))
        )

    return (
        dominant_frequency,
        weighted_frequency,
        spectral_entropy
    )


# ============================================================
# 3. CREATE FREQUENCY FEATURES FOR WINDOWS
# ============================================================

def create_frequency_features(
    df,
    signal_columns,
    sampling_rate
):
    """
    Create frequency features for every window.

    One output row = one window.
    """

    feature_rows = []

    grouped = df.groupby(
        "window_id",
        sort=False
    )

    for window_id, window in grouped:

        features = {}

        # ----------------------------------------------------
        # Window identifier
        # ----------------------------------------------------

        features["window_id"] = (
            window_id
        )

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

                features[column] = (
                    window[column].iloc[0]
                )

        # ----------------------------------------------------
        # Calculate frequency features
        # ----------------------------------------------------

        for signal in signal_columns:

            if signal not in window.columns:
                continue

            values = pd.to_numeric(
                window[signal],
                errors="coerce"
            ).dropna()

            if len(values) < 2:
                continue

            (
                dominant_frequency,
                weighted_frequency,
                spectral_entropy
            ) = calculate_frequency_features(
                values=values.to_numpy(),
                sampling_rate=sampling_rate
            )

            features[
                f"{signal}_dominant_frequency"
            ] = dominant_frequency

            features[
                f"{signal}_weighted_frequency"
            ] = weighted_frequency

            features[
                f"{signal}_spectral_entropy"
            ] = spectral_entropy

        feature_rows.append(
            features
        )

    return pd.DataFrame(
        feature_rows
    )


# ============================================================
# 4. ACCELEROMETER
# ============================================================

def process_accelerometer():

    input_file = (
        PROCESSED_DIR /
        "windowed_accelerometer.csv"
    )

    output_file = (
        FEATURES_DIR /
        "frequency_features_accelerometer.csv"
    )

    print("\n" + "=" * 60)
    print("ACCELEROMETER FREQUENCY FEATURES")
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

    # --------------------------------------------------------
    # Create filtered acceleration magnitude
    # --------------------------------------------------------

    required_columns = [
        "ax_filtered",
        "ay_filtered",
        "az_filtered"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    df["acc_magnitude_filtered"] = np.sqrt(
        df["ax_filtered"] ** 2
        +
        df["ay_filtered"] ** 2
        +
        df["az_filtered"] ** 2
    )

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    signal_columns = [
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
        "acc_magnitude_filtered"
    ]

    result = create_frequency_features(
        df=df,
        signal_columns=signal_columns,
        sampling_rate=12.5
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

        if (
            "dominant_frequency" in column
            or "weighted_frequency" in column
            or "spectral_entropy" in column
        ):

            print(
                f"  - {column}"
            )

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# 5. GYROSCOPE
# ============================================================

def process_gyroscope():

    input_file = (
        PROCESSED_DIR /
        "windowed_gyroscope.csv"
    )

    output_file = (
        FEATURES_DIR /
        "frequency_features_gyroscope.csv"
    )

    print("\n" + "=" * 60)
    print("GYROSCOPE FREQUENCY FEATURES")
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

    # --------------------------------------------------------
    # Create filtered gyroscope magnitude
    # --------------------------------------------------------

    required_columns = [
        "gx_filtered",
        "gy_filtered",
        "gz_filtered"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    df["gyro_magnitude_filtered"] = np.sqrt(
        df["gx_filtered"] ** 2
        +
        df["gy_filtered"] ** 2
        +
        df["gz_filtered"] ** 2
    )

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    signal_columns = [
        "gx_filtered",
        "gy_filtered",
        "gz_filtered",
        "gyro_magnitude_filtered"
    ]

    result = create_frequency_features(
        df=df,
        signal_columns=signal_columns,
        sampling_rate=25.0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

        if (
            "dominant_frequency" in column
            or "weighted_frequency" in column
            or "spectral_entropy" in column
        ):

            print(
                f"  - {column}"
            )

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - FREQUENCY FEATURES")
    print("=" * 60)

    process_accelerometer()

    process_gyroscope()

    print("\n" + "=" * 60)
    print("FREQUENCY FEATURES COMPLETED ✅")
    print("=" * 60)


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":

    main()