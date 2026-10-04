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
# 2. MAGNITUDE FUNCTION
# ============================================================

def calculate_magnitude(x, y, z):
    """
    Calculate 3-axis sensor magnitude.

    Formula:

        magnitude = sqrt(x^2 + y^2 + z^2)
    """

    return np.sqrt(
        x**2 +
        y**2 +
        z**2
    )


# ============================================================
# 3. ACCELEROMETER MAGNITUDE
# ============================================================

def process_accelerometer():

    input_file = (
        PROCESSED_DIR /
        "windowed_accelerometer.csv"
    )

    output_file = (
        FEATURES_DIR /
        "magnitude_accelerometer.csv"
    )

    print("\n" + "=" * 60)
    print("ACCELEROMETER MAGNITUDE")
    print("=" * 60)

    if not input_file.exists():

        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_csv(input_file)

    print(
        f"\nInput shape: {df.shape}"
    )

    required_columns = [
        "ax",
        "ay",
        "az",
        "ax_filtered",
        "ay_filtered",
        "az_filtered"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------------
    # Raw magnitude
    # --------------------------------------------------------

    df["acc_magnitude"] = calculate_magnitude(
        df["ax"],
        df["ay"],
        df["az"]
    )

    # --------------------------------------------------------
    # Filtered magnitude
    # --------------------------------------------------------

    df["acc_magnitude_filtered"] = calculate_magnitude(
        df["ax_filtered"],
        df["ay_filtered"],
        df["az_filtered"]
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nCreated features:"
    )

    print(
        "  acc_magnitude"
    )

    print(
        "  acc_magnitude_filtered"
    )

    print(
        f"\nOutput shape: {df.shape}"
    )

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# 4. GYROSCOPE MAGNITUDE
# ============================================================

def process_gyroscope():

    input_file = (
        PROCESSED_DIR /
        "windowed_gyroscope.csv"
    )

    output_file = (
        FEATURES_DIR /
        "magnitude_gyroscope.csv"
    )

    print("\n" + "=" * 60)
    print("GYROSCOPE MAGNITUDE")
    print("=" * 60)

    if not input_file.exists():

        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_csv(input_file)

    print(
        f"\nInput shape: {df.shape}"
    )

    required_columns = [
        "gx",
        "gy",
        "gz",
        "gx_filtered",
        "gy_filtered",
        "gz_filtered"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------------
    # Raw magnitude
    # --------------------------------------------------------

    df["gyro_magnitude"] = calculate_magnitude(
        df["gx"],
        df["gy"],
        df["gz"]
    )

    # --------------------------------------------------------
    # Filtered magnitude
    # --------------------------------------------------------

    df["gyro_magnitude_filtered"] = calculate_magnitude(
        df["gx_filtered"],
        df["gy_filtered"],
        df["gz_filtered"]
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nCreated features:"
    )

    print(
        "  gyro_magnitude"
    )

    print(
        "  gyro_magnitude_filtered"
    )

    print(
        f"\nOutput shape: {df.shape}"
    )

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - MAGNITUDE FEATURES")
    print("=" * 60)

    process_accelerometer()

    process_gyroscope()

    print("\n" + "=" * 60)
    print("MAGNITUDE FEATURES COMPLETED ✅")
    print("=" * 60)


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":
    main()