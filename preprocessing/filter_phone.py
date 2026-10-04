from pathlib import Path

import pandas as pd
from scipy.signal import butter, filtfilt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


ACC_INPUT = PROCESSED_DIR / "phone_accelerometer_12_5hz.csv"
GYRO_INPUT = PROCESSED_DIR / "phone_gyroscope_25hz.csv"

ACC_OUTPUT = PROCESSED_DIR / "phone_accelerometer_filtered.csv"
GYRO_OUTPUT = PROCESSED_DIR / "phone_gyroscope_filtered.csv"


# ============================================================
# FILTER SETTINGS
# ============================================================

CUTOFF_FREQUENCY = 1.3

ACCELEROMETER_FS = 12.5
GYROSCOPE_FS = 25.0

FILTER_ORDER = 4


# ============================================================
# FILTER FUNCTION
# ============================================================

def apply_filter(values, fs):
    """
    Apply 4th-order Butterworth low-pass filter.
    """

    nyquist = fs / 2
    normalized_cutoff = CUTOFF_FREQUENCY / nyquist

    b, a = butter(
        FILTER_ORDER,
        normalized_cutoff,
        btype="lowpass"
    )

    minimum_samples = max(len(a), len(b)) * 3

    if len(values) < minimum_samples:
        raise ValueError(
            f"Not enough samples for filtering. "
            f"Need at least {minimum_samples}, "
            f"got {len(values)}."
        )

    return filtfilt(
        b,
        a,
        values.astype(float)
    )


# ============================================================
# ACCELEROMETER
# ============================================================

def filter_accelerometer():

    print("\n" + "=" * 60)
    print("PHONE ACCELEROMETER FILTERING")
    print("=" * 60)

    df = pd.read_csv(ACC_INPUT)

    print("Input shape:", df.shape)

    for column in ["ax", "ay", "az"]:

        filtered_parts = []

        for session_id, group in df.groupby(
            "session_id",
            sort=False
        ):

            values = group[column].reset_index(drop=True)

            filtered = apply_filter(
                values,
                ACCELEROMETER_FS
            )

            temp = group.copy()
            temp[f"{column}_filtered"] = filtered

            filtered_parts.append(temp)

        df = pd.concat(
            filtered_parts,
            ignore_index=True
        )

    df.to_csv(
        ACC_OUTPUT,
        index=False
    )

    print("Output shape:", df.shape)
    print("Saved:", ACC_OUTPUT)


# ============================================================
# GYROSCOPE
# ============================================================

def filter_gyroscope():

    print("\n" + "=" * 60)
    print("PHONE GYROSCOPE FILTERING")
    print("=" * 60)

    df = pd.read_csv(GYRO_INPUT)

    print("Input shape:", df.shape)

    for column in ["gx", "gy", "gz"]:

        filtered_parts = []

        for session_id, group in df.groupby(
            "session_id",
            sort=False
        ):

            values = group[column].reset_index(drop=True)

            filtered = apply_filter(
                values,
                GYROSCOPE_FS
            )

            temp = group.copy()
            temp[f"{column}_filtered"] = filtered

            filtered_parts.append(temp)

        df = pd.concat(
            filtered_parts,
            ignore_index=True
        )

    df.to_csv(
        GYRO_OUTPUT,
        index=False
    )

    print("Output shape:", df.shape)
    print("Saved:", GYRO_OUTPUT)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - PHONE FILTERING")
    print("=" * 60)

    print(f"\nCutoff frequency: {CUTOFF_FREQUENCY} Hz")
    print(f"Filter order: {FILTER_ORDER}")

    filter_accelerometer()
    filter_gyroscope()

    print("\n" + "=" * 60)
    print("PHONE FILTERING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()