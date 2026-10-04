from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
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
    "phone_frequency_features_accelerometer.csv"
)

GYRO_OUTPUT = (
    PROCESSED_DIR /
    "phone_frequency_features_gyroscope.csv"
)


# ============================================================
# 2. FREQUENCY FEATURE FUNCTION
# ============================================================

def calculate_frequency_features(
    values,
    sampling_rate
):
    """
    Calculate:

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

    values = values - np.mean(values)

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    fft_values = np.fft.rfft(values)

    frequencies = np.fft.rfftfreq(
        len(values),
        d=1 / sampling_rate
    )

    # --------------------------------------------------------
    # Power spectrum
    # --------------------------------------------------------

    power = np.abs(
        fft_values
    ) ** 2

    # Remove zero-frequency component
    if len(power) > 1:

        frequencies = frequencies[1:]
        power = power[1:]

    total_power = np.sum(power)

    if total_power <= 0:

        return 0.0, 0.0, 0.0

    # ========================================================
    # 1. DOMINANT FREQUENCY
    # ========================================================

    dominant_index = np.argmax(power)

    dominant_frequency = (
        frequencies[dominant_index]
    )

    # ========================================================
    # 2. WEIGHTED FREQUENCY
    # ========================================================

    weighted_frequency = (
        np.sum(frequencies * power)
        / total_power
    )

    # ========================================================
    # 3. SPECTRAL ENTROPY
    # ========================================================

    probability = (
        power / total_power
    )

    probability = probability[
        probability > 0
    ]

    spectral_entropy = -np.sum(
        probability *
        np.log2(probability)
    )

    # Normalize between 0 and 1
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
# 3. CREATE WINDOW FEATURES
# ============================================================

def create_frequency_features(
    df,
    signal_columns,
    sampling_rate
):

    rows = []

    grouped = df.groupby(
        "window_id",
        sort=False
    )

    for window_id, window in grouped:

        features = {}

        features["window_id"] = window_id

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
                values.to_numpy(),
                sampling_rate
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

        rows.append(features)

    return pd.DataFrame(rows)


# ============================================================
# 4. ACCELEROMETER
# ============================================================

def process_accelerometer():

    print("\n" + "=" * 60)
    print("PHONE ACCELEROMETER FREQUENCY FEATURES")
    print("=" * 60)

    df = pd.read_csv(
        ACC_INPUT
    )

    print(
        "Input shape:",
        df.shape
    )

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

    result = create_frequency_features(
        df=df,
        signal_columns=signal_columns,
        sampling_rate=12.5
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

    print("\nCreated frequency features:")

    for col in result.columns:
        if col != "window_id":
            print(" ", col)

    print("\nSaved:")
    print(ACC_OUTPUT)


# ============================================================
# 5. GYROSCOPE
# ============================================================

def process_gyroscope():

    print("\n" + "=" * 60)
    print("PHONE GYROSCOPE FREQUENCY FEATURES")
    print("=" * 60)

    df = pd.read_csv(
        GYRO_INPUT
    )

    print(
        "Input shape:",
        df.shape
    )

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

    result = create_frequency_features(
        df=df,
        signal_columns=signal_columns,
        sampling_rate=25.0
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

    print("\nCreated frequency features:")

    for col in result.columns:
        if col != "window_id":
            print(" ", col)

    print("\nSaved:")
    print(GYRO_OUTPUT)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - PHONE FREQUENCY FEATURES")
    print("=" * 60)

    process_accelerometer()

    process_gyroscope()

    print("\n" + "=" * 60)
    print("PHONE FREQUENCY FEATURES COMPLETED")
    print("=" * 60)


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":
    main()