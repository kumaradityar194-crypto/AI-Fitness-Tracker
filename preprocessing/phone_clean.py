from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "phone_squat_fresh.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_imu_cleaned.csv"
)


GRAVITY_MS2_PER_G = 9.80665
RAD_TO_DEG = 180.0 / np.pi


print("=" * 70)
print("AI FITNESS TRACKER - PHONE IMU CLEANING")
print("=" * 70)

print()
print(f"Input file: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Phone IMU input file not found: {INPUT_FILE}"
    )

print()
print("Loading phone IMU data...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Original shape: {df.shape}"
)


required_columns = [
    "timestamp_ms",
    "sensor_timestamp_ns",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "device_id",
    "source",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


df = df[
    required_columns
].copy()


numeric_columns = [
    "timestamp_ms",
    "sensor_timestamp_ns",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]


for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


before_rows = len(df)

df = df.dropna(
    subset=numeric_columns
).copy()

removed_invalid = (
    before_rows
    - len(df)
)

print()
print(
    f"Removed invalid rows: {removed_invalid}"
)


before_rows = len(df)

df = df.drop_duplicates(
    subset=["timestamp_ms"],
    keep="first",
).copy()

removed_duplicates = (
    before_rows
    - len(df)
)

print(
    f"Removed duplicate timestamps: "
    f"{removed_duplicates}"
)


df = (
    df
    .sort_values(
        "timestamp_ms"
    )
    .reset_index(
        drop=True
    )
)


if df.empty:
    raise ValueError(
        "No valid IMU rows remain after cleaning."
    )


accelerometer_columns = [
    "ax",
    "ay",
    "az",
]

gyroscope_columns = [
    "gx",
    "gy",
    "gz",
]


# ============================================================
# AUTOMATIC ACCELEROMETER UNIT DETECTION
# ============================================================

accelerometer_magnitude = np.sqrt(
    df["ax"].to_numpy(dtype=float) ** 2
    + df["ay"].to_numpy(dtype=float) ** 2
    + df["az"].to_numpy(dtype=float) ** 2
)

accelerometer_reference = float(
    np.median(
        accelerometer_magnitude
    )
)


if accelerometer_reference > 3.0:

    accelerometer_unit = "m/s^2"

    print()
    print(
        "Accelerometer unit detected: m/s^2"
    )

    print(
        "Accelerometer conversion: m/s^2 to g"
    )

    for column in accelerometer_columns:
        df[column] = (
            df[column]
            / GRAVITY_MS2_PER_G
        )

else:

    accelerometer_unit = "g"

    print()
    print(
        "Accelerometer unit detected: g"
    )

    print(
        "Accelerometer conversion: none"
    )


# ============================================================
# AUTOMATIC GYROSCOPE UNIT DETECTION
# ============================================================

gyro_absolute_values = np.concatenate(
    [
        np.abs(
            df[column].to_numpy(
                dtype=float
            )
        )
        for column in gyroscope_columns
    ]
)

gyro_percentile_95 = float(
    np.percentile(
        gyro_absolute_values,
        95,
    )
)


if gyro_percentile_95 > 100.0:

    gyroscope_unit = "deg/s"

    print()
    print(
        "Gyroscope unit detected: deg/s"
    )

    print(
        "Gyroscope conversion: none"
    )

elif gyro_percentile_95 < 10.0:

    gyroscope_unit = "rad/s"

    print()
    print(
        "Gyroscope unit detected: rad/s"
    )

    print(
        "Gyroscope conversion: rad/s to deg/s"
    )

    for column in gyroscope_columns:
        df[column] = (
            df[column]
            * RAD_TO_DEG
        )

else:

    gyroscope_unit = "deg/s"

    print()
    print(
        "Gyroscope unit detected: deg/s"
    )

    print(
        "Gyroscope conversion: none"
    )


# ============================================================
# TIME
# ============================================================

df["time_s"] = (
    df["timestamp_ms"]
    - df["timestamp_ms"].iloc[0]
) / 1000.0


# ============================================================
# SUMMARY
# ============================================================

print()
print("Detected sensor units:")
print(
    f"Accelerometer: {accelerometer_unit}"
)
print(
    f"Gyroscope: {gyroscope_unit}"
)


print()
print("Normalized accelerometer range:")

for column in accelerometer_columns:
    print(
        f"{column}: "
        f"{df[column].min():.6f} "
        f"to "
        f"{df[column].max():.6f} g"
    )


print()
print("Normalized gyroscope range:")

for column in gyroscope_columns:
    print(
        f"{column}: "
        f"{df[column].min():.6f} "
        f"to "
        f"{df[column].max():.6f} deg/s"
    )


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print(
    f"Final shape: {df.shape}"
)

print(
    f"Time duration: "
    f"{df['time_s'].iloc[-1]:.2f} seconds"
)

print()
print("Final columns:")
print(
    df.columns.tolist()
)

print()
print("First 5 normalized rows:")
print(
    df.head().to_string(
        index=False
    )
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print("=" * 70)
print("PHONE IMU CLEANING COMPLETED")
print("=" * 70)

