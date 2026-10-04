from pathlib import Path

import pandas as pd
from scipy.signal import butter, filtfilt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# 2. FILTER SETTINGS
# ============================================================

# Paper-inspired cutoff frequency
CUTOFF_FREQUENCY = 1.3

# Sensor sampling frequencies
ACCELEROMETER_FS = 12.5
GYROSCOPE_FS = 25.0

# Butterworth filter order
FILTER_ORDER = 4


# ============================================================
# 3. CREATE BUTTERWORTH LOW-PASS FILTER
# ============================================================

def create_lowpass_filter(fs, cutoff, order=4):
    """
    Create Butterworth low-pass filter coefficients.

    fs     -> sampling frequency
    cutoff -> cutoff frequency
    order  -> filter order
    """

    nyquist = fs / 2

    normalized_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normalized_cutoff,
        btype="lowpass"
    )

    return b, a


# ============================================================
# 4. APPLY FILTER
# ============================================================

def apply_filter(series, fs):
    """
    Apply zero-phase Butterworth low-pass filtering
    to one sensor signal.
    """

    b, a = create_lowpass_filter(
        fs=fs,
        cutoff=CUTOFF_FREQUENCY,
        order=FILTER_ORDER
    )

    values = series.astype(float).to_numpy()

    # filtfilt needs enough samples
    minimum_samples = max(len(a), len(b)) * 3

    if len(values) < minimum_samples:
        raise ValueError(
            f"Not enough samples for filtering. "
            f"Need at least {minimum_samples}, "
            f"but got {len(values)}."
        )

    filtered_values = filtfilt(
        b,
        a,
        values
    )

    return filtered_values


# ============================================================
# 5. FILTER ACCELEROMETER
# ============================================================

def filter_accelerometer():

    input_file = (
        PROCESSED_DIR /
        "cleaned_accelerometer.csv"
    )

    output_file = (
        PROCESSED_DIR /
        "filtered_accelerometer.csv"
    )

    print("\n" + "=" * 60)
    print("FILTERING ACCELEROMETER")
    print("=" * 60)

    if not input_file.exists():
        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_csv(input_file)

    print(f"\nInput shape: {df.shape}")

    sensor_columns = [
        "ax",
        "ay",
        "az"
    ]

    for column in sensor_columns:

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found "
                f"in accelerometer dataset."
            )

        print(f"Filtering: {column}")

        df[f"{column}_filtered"] = apply_filter(
            df[column],
            ACCELEROMETER_FS
        )

    df.to_csv(
        output_file,
        index=False
    )

    print(f"\nOutput shape: {df.shape}")

    print(
        f"\nSaved filtered accelerometer data:\n"
        f"{output_file}"
    )


# ============================================================
# 6. FILTER GYROSCOPE
# ============================================================

def filter_gyroscope():

    input_file = (
        PROCESSED_DIR /
        "cleaned_gyroscope.csv"
    )

    output_file = (
        PROCESSED_DIR /
        "filtered_gyroscope.csv"
    )

    print("\n" + "=" * 60)
    print("FILTERING GYROSCOPE")
    print("=" * 60)

    if not input_file.exists():
        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_csv(input_file)

    print(f"\nInput shape: {df.shape}")

    sensor_columns = [
        "gx",
        "gy",
        "gz"
    ]

    for column in sensor_columns:

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found "
                f"in gyroscope dataset."
            )

        print(f"Filtering: {column}")

        df[f"{column}_filtered"] = apply_filter(
            df[column],
            GYROSCOPE_FS
        )

    df.to_csv(
        output_file,
        index=False
    )

    print(f"\nOutput shape: {df.shape}")

    print(
        f"\nSaved filtered gyroscope data:\n"
        f"{output_file}"
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - SIGNAL FILTERING")
    print("=" * 60)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nProcessed folder:")
    print(PROCESSED_DIR)

    print(
        f"\nFilter type: "
        f"Butterworth Low-Pass"
    )

    print(
        f"Cutoff frequency: "
        f"{CUTOFF_FREQUENCY} Hz"
    )

    print(
        f"Accelerometer sampling rate: "
        f"{ACCELEROMETER_FS} Hz"
    )

    print(
        f"Gyroscope sampling rate: "
        f"{GYROSCOPE_FS} Hz"
    )

    print(
        f"Filter order: "
        f"{FILTER_ORDER}"
    )

    filter_accelerometer()

    filter_gyroscope()

    print("\n" + "=" * 60)
    print("SIGNAL FILTERING COMPLETED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()