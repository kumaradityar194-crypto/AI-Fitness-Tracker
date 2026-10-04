from pathlib import Path
import pandas as pd
import re


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FIND CSV FILES
# ============================================================

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in:\n{RAW_DIR}"
    )

print("=" * 60)
print("AI FITNESS TRACKER - DATA CLEANING")
print("=" * 60)

print(f"\nProject root:\n{PROJECT_ROOT}")
print(f"\nRaw folder:\n{RAW_DIR}")
print(f"\nProcessed folder:\n{PROCESSED_DIR}")
print(f"\nTotal CSV files found: {len(csv_files)}")


# ============================================================
# 3. CLASSIFY SENSOR FILE
# ============================================================

def get_sensor_type(filename):
    """
    Detect whether a file contains accelerometer
    or gyroscope data.
    """

    name = filename.lower()

    if "accelerometer" in name:
        return "accelerometer"

    if "gyroscope" in name:
        return "gyroscope"

    return "unknown"


# ============================================================
# 4. EXTRACT METADATA FROM FILE NAME
# ============================================================

def extract_metadata(filename):

    name = Path(filename).stem

    # --------------------------------------------------------
    # Participant
    # Example: A-bench-heavy...
    # --------------------------------------------------------

    parts = name.split("-")

    participant = parts[0]

    # --------------------------------------------------------
    # Exercise
    # --------------------------------------------------------

    exercise_mapping = {
        "bench": "bench",
        "dead": "deadlift",
        "ohp": "ohp",
        "row": "row",
        "squat": "squat",
        "rest": "rest"
    }

    exercise = "unknown"

    if len(parts) > 1:
        raw_exercise = parts[1].lower()

        exercise = exercise_mapping.get(
            raw_exercise,
            raw_exercise
        )

    # --------------------------------------------------------
    # Intensity
    # --------------------------------------------------------

    intensity = "unknown"

    lower_name = name.lower()

    if "heavy" in lower_name:
        intensity = "heavy"

    elif "medium" in lower_name:
        intensity = "medium"

    elif "sitting" in lower_name:
        intensity = "sitting"

    elif "standing" in lower_name:
        intensity = "standing"

    elif "rest" in lower_name:
        intensity = "rest"

    # --------------------------------------------------------
    # RPE
    # --------------------------------------------------------

    rpe = None

    rpe_match = re.search(r"rpe(\d+)", lower_name)

    if rpe_match:
        rpe = int(rpe_match.group(1))

    return participant, exercise, intensity, rpe


# ============================================================
# 5. READ SENSOR FILES
# ============================================================

accelerometer_data = []
gyroscope_data = []

for file in csv_files:

    sensor_type = get_sensor_type(file.name)

    if sensor_type == "unknown":
        print(f"Skipping unknown file: {file.name}")
        continue

    print(f"Reading: {file.name}")

    df = pd.read_csv(file)

    participant, exercise, intensity, rpe = (
        extract_metadata(file.name)
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    df["participant"] = participant
    df["exercise"] = exercise
    df["intensity"] = intensity
    df["rpe"] = rpe
    df["source_file"] = file.name

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    if "epoch (ms)" not in df.columns:

        print(
            f"Skipping {file.name}: "
            "epoch (ms) column not found."
        )

        continue

    df["epoch (ms)"] = pd.to_numeric(
        df["epoch (ms)"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["epoch (ms)"]
    )

    # --------------------------------------------------------
    # Convert sensor values
    # --------------------------------------------------------

    if sensor_type == "accelerometer":

        sensor_columns = [
            "x-axis (g)",
            "y-axis (g)",
            "z-axis (g)"
        ]

        for col in sensor_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # Rename for standard project format

        rename_dict = {
            "x-axis (g)": "ax",
            "y-axis (g)": "ay",
            "z-axis (g)": "az"
        }

        df = df.rename(
            columns=rename_dict
        )

        accelerometer_data.append(df)

    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    elif sensor_type == "gyroscope":

        sensor_columns = [
            "x-axis (deg/s)",
            "y-axis (deg/s)",
            "z-axis (deg/s)"
        ]

        for col in sensor_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # Rename for standard project format

        rename_dict = {
            "x-axis (deg/s)": "gx",
            "y-axis (deg/s)": "gy",
            "z-axis (deg/s)": "gz"
        }

        df = df.rename(
            columns=rename_dict
        )

        gyroscope_data.append(df)


# ============================================================
# 6. COMBINE SAME SENSOR FILES
# ============================================================

print("\n" + "=" * 60)
print("COMBINING SENSOR DATA")
print("=" * 60)

if accelerometer_data:

    acc_df = pd.concat(
        accelerometer_data,
        ignore_index=True
    )

else:

    acc_df = pd.DataFrame()


if gyroscope_data:

    gyro_df = pd.concat(
        gyroscope_data,
        ignore_index=True
    )

else:

    gyro_df = pd.DataFrame()


print(
    f"Accelerometer rows: {len(acc_df)}"
)

print(
    f"Gyroscope rows: {len(gyro_df)}"
)


# ============================================================
# 7. REMOVE DUPLICATES INSIDE EACH SENSOR STREAM
# ============================================================

if not acc_df.empty:

    acc_df = acc_df.drop_duplicates(
        subset=[
            "epoch (ms)",
            "participant",
            "exercise",
            "source_file"
        ]
    )

if not gyro_df.empty:

    gyro_df = gyro_df.drop_duplicates(
        subset=[
            "epoch (ms)",
            "participant",
            "exercise",
            "source_file"
        ]
    )


# ============================================================
# 8. SORT BY TIMESTAMP
# ============================================================

if not acc_df.empty:

    acc_df = acc_df.sort_values(
        "epoch (ms)"
    )

if not gyro_df.empty:

    gyro_df = gyro_df.sort_values(
        "epoch (ms)"
    )


# ============================================================
# 9. SAVE CLEAN ACCELEROMETER DATA
# ============================================================

if not acc_df.empty:

    acc_output = (
        PROCESSED_DIR /
        "cleaned_accelerometer.csv"
    )

    acc_df.to_csv(
        acc_output,
        index=False
    )

    print(
        f"\nSaved:\n{acc_output}"
    )


# ============================================================
# 10. SAVE CLEAN GYROSCOPE DATA
# ============================================================

if not gyro_df.empty:

    gyro_output = (
        PROCESSED_DIR /
        "cleaned_gyroscope.csv"
    )

    gyro_df.to_csv(
        gyro_output,
        index=False
    )

    print(
        f"Saved:\n{gyro_output}"
    )


# ============================================================
# 11. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

if not acc_df.empty:

    print("\nAccelerometer shape:")
    print(acc_df.shape)

    print("\nAccelerometer exercises:")

    print(
        acc_df["exercise"]
        .value_counts()
    )

    print("\nAccelerometer participants:")

    print(
        acc_df["participant"]
        .value_counts()
    )

    print("\nAccelerometer intensity:")

    print(
        acc_df["intensity"]
        .value_counts()
    )


if not gyro_df.empty:

    print("\nGyroscope shape:")
    print(gyro_df.shape)

    print("\nGyroscope exercises:")

    print(
        gyro_df["exercise"]
        .value_counts()
    )


print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)