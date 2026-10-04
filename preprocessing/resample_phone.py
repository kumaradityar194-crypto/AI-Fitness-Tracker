import pandas as pd
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/phone_imu_sessions.csv"
)

ACC_OUTPUT = Path(
    "data/processed/phone_accelerometer_12_5hz.csv"
)

GYRO_OUTPUT = Path(
    "data/processed/phone_gyroscope_25hz.csv"
)


def resample_session(group, interval_ms):
    """
    Resample one recording session to a fixed time interval.
    """

    group = group.copy()

    # Use timestamp as time index
    group["time"] = pd.to_datetime(
        group["timestamp_ms"],
        unit="ms"
    )

    group = group.set_index("time")

    # Numeric sensor columns only
    sensor_columns = [
        "ax", "ay", "az",
        "gx", "gy", "gz"
    ]

    result = (
        group[sensor_columns]
        .resample(f"{interval_ms}ms")
        .mean()
        .interpolate(method="linear")
    )

    # Remove remaining missing values
    result = result.dropna().reset_index()

    return result


# --------------------------------------------------
# Load
# --------------------------------------------------

print("=" * 60)
print("PHONE IMU RESAMPLING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:", df.shape)


# --------------------------------------------------
# Accelerometer → 12.5 Hz
# 12.5 Hz = 80 ms interval
# --------------------------------------------------

acc_parts = []

for session_id, group in df.groupby("session_id"):

    print(
        f"\nAccelerometer - Session {session_id}"
    )

    resampled = resample_session(
        group,
        interval_ms=80
    )

    resampled["session_id"] = session_id

    acc_parts.append(resampled)

acc_df = pd.concat(
    acc_parts,
    ignore_index=True
)

acc_df.to_csv(
    ACC_OUTPUT,
    index=False
)

print(
    "\nAccelerometer output:",
    acc_df.shape
)


# --------------------------------------------------
# Gyroscope → 25 Hz
# 25 Hz = 40 ms interval
# --------------------------------------------------

gyro_parts = []

for session_id, group in df.groupby("session_id"):

    print(
        f"\nGyroscope - Session {session_id}"
    )

    resampled = resample_session(
        group,
        interval_ms=40
    )

    resampled["session_id"] = session_id

    gyro_parts.append(resampled)

gyro_df = pd.concat(
    gyro_parts,
    ignore_index=True
)

gyro_df.to_csv(
    GYRO_OUTPUT,
    index=False
)

print(
    "\nGyroscope output:",
    gyro_df.shape
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("RESAMPLING COMPLETED")
print("=" * 60)

print("\nAccelerometer:")
print(ACC_OUTPUT)

print("\nGyroscope:")
print(GYRO_OUTPUT)

print("\nAccelerometer samples:", len(acc_df))
print("Gyroscope samples:", len(gyro_df))