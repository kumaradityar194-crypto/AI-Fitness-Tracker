from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


ACC_INPUT = (
    PROCESSED_DIR /
    "phone_accelerometer_windowed.csv"
)

GYRO_INPUT = (
    PROCESSED_DIR /
    "phone_gyroscope_windowed.csv"
)

ACC_OUTPUT = (
    PROCESSED_DIR /
    "phone_accelerometer_magnitude.csv"
)

GYRO_OUTPUT = (
    PROCESSED_DIR /
    "phone_gyroscope_magnitude.csv"
)


# ============================================================
# MAGNITUDE FUNCTION
# ============================================================

def calculate_magnitude(x, y, z):

    return np.sqrt(
        x**2 +
        y**2 +
        z**2
    )


# ============================================================
# ACCELEROMETER
# ============================================================

def process_accelerometer():

    print("\n" + "=" * 60)
    print("PHONE ACCELEROMETER MAGNITUDE")
    print("=" * 60)

    df = pd.read_csv(ACC_INPUT)

    print("Input shape:", df.shape)

    required = [
        "ax",
        "ay",
        "az",
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
        "window_id"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing accelerometer columns: {missing}"
        )

    # Raw magnitude
    df["acc_magnitude"] = calculate_magnitude(
        df["ax"],
        df["ay"],
        df["az"]
    )

    # Filtered magnitude
    df["acc_magnitude_filtered"] = calculate_magnitude(
        df["ax_filtered"],
        df["ay_filtered"],
        df["az_filtered"]
    )

    df.to_csv(
        ACC_OUTPUT,
        index=False
    )

    print("Created:")
    print("  acc_magnitude")
    print("  acc_magnitude_filtered")

    print("Output shape:", df.shape)

    print("Windows:", df["window_id"].nunique())

    print("Saved:", ACC_OUTPUT)


# ============================================================
# GYROSCOPE
# ============================================================

def process_gyroscope():

    print("\n" + "=" * 60)
    print("PHONE GYROSCOPE MAGNITUDE")
    print("=" * 60)

    df = pd.read_csv(GYRO_INPUT)

    print("Input shape:", df.shape)

    required = [
        "gx",
        "gy",
        "gz",
        "gx_filtered",
        "gy_filtered",
        "gz_filtered",
        "window_id"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing gyroscope columns: {missing}"
        )

    # Raw magnitude
    df["gyro_magnitude"] = calculate_magnitude(
        df["gx"],
        df["gy"],
        df["gz"]
    )

    # Filtered magnitude
    df["gyro_magnitude_filtered"] = calculate_magnitude(
        df["gx_filtered"],
        df["gy_filtered"],
        df["gz_filtered"]
    )

    df.to_csv(
        GYRO_OUTPUT,
        index=False
    )

    print("Created:")
    print("  gyro_magnitude")
    print("  gyro_magnitude_filtered")

    print("Output shape:", df.shape)

    print("Windows:", df["window_id"].nunique())

    print("Saved:", GYRO_OUTPUT)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - PHONE MAGNITUDE")
    print("=" * 60)

    process_accelerometer()

    process_gyroscope()

    print("\n" + "=" * 60)
    print("PHONE MAGNITUDE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()