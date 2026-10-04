from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# 2. WINDOW SETTINGS
# ============================================================

WINDOW_SECONDS = 4

OVERLAP = 0.50


# Native sampling rates of our sensor files
ACCELEROMETER_FS = 12.5
GYROSCOPE_FS = 25.0


# ============================================================
# 3. CREATE WINDOWS
# ============================================================

def create_windows(
    df,
    sensor_columns,
    sampling_rate
):
    """
    Divide one sensor recording into fixed-size windows.

    Important:
    Windows are created separately for every source file,
    so two different recordings are never mixed.
    """

    samples_per_window = int(
        WINDOW_SECONDS * sampling_rate
    )

    step_size = int(
        samples_per_window * (1 - OVERLAP)
    )

    print(
        f"\nSampling rate: {sampling_rate} Hz"
    )

    print(
        f"Samples per window: "
        f"{samples_per_window}"
    )

    print(
        f"Step size: "
        f"{step_size}"
    )

    windowed_data = []

    window_id = 0

    # --------------------------------------------------------
    # Process every original recording separately
    # --------------------------------------------------------

    for source_file, recording in df.groupby(
        "source_file",
        sort=False
    ):

        recording = recording.reset_index(
            drop=True
        )

        total_samples = len(recording)

        print(
            f"\nProcessing: {source_file}"
        )

        print(
            f"Total samples: {total_samples}"
        )

        # ----------------------------------------------------
        # Create overlapping windows
        # ----------------------------------------------------

        for start in range(
            0,
            total_samples - samples_per_window + 1,
            step_size
        ):

            end = (
                start +
                samples_per_window
            )

            window = recording.iloc[
                start:end
            ].copy()

            # ------------------------------------------------
            # Add window metadata
            # ------------------------------------------------

            window["window_id"] = window_id

            window["window_sample"] = range(
                len(window)
            )

            window["window_start_s"] = (
                window["epoch (ms)"].iloc[0]
                - recording["epoch (ms)"].iloc[0]
            ) / 1000.0

            window["window_end_s"] = (
                window["epoch (ms)"].iloc[-1]
                - recording["epoch (ms)"].iloc[0]
            ) / 1000.0

            windowed_data.append(
                window
            )

            window_id += 1

    if not windowed_data:

        raise ValueError(
            "No windows could be created. "
            "Check the recording length."
        )

    result = pd.concat(
        windowed_data,
        ignore_index=True
    )

    return result


# ============================================================
# 4. ACCELEROMETER WINDOWING
# ============================================================

def process_accelerometer():

    input_file = (
        PROCESSED_DIR /
        "filtered_accelerometer.csv"
    )

    output_file = (
        PROCESSED_DIR /
        "windowed_accelerometer.csv"
    )

    print("\n" + "=" * 60)
    print("WINDOWING ACCELEROMETER")
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

    required_columns = [
        "epoch (ms)",
        "ax",
        "ay",
        "az",
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
        "participant",
        "exercise",
        "intensity",
        "source_file"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: "
                f"{column}"
            )

    result = create_windows(
        df=df,
        sensor_columns=[
            "ax",
            "ay",
            "az"
        ],
        sampling_rate=ACCELEROMETER_FS
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nTotal windows: "
        f"{result['window_id'].nunique()}"
    )

    print(
        f"Output shape: {result.shape}"
    )

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# 5. GYROSCOPE WINDOWING
# ============================================================

def process_gyroscope():

    input_file = (
        PROCESSED_DIR /
        "filtered_gyroscope.csv"
    )

    output_file = (
        PROCESSED_DIR /
        "windowed_gyroscope.csv"
    )

    print("\n" + "=" * 60)
    print("WINDOWING GYROSCOPE")
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

    required_columns = [
        "epoch (ms)",
        "gx",
        "gy",
        "gz",
        "gx_filtered",
        "gy_filtered",
        "gz_filtered",
        "participant",
        "exercise",
        "intensity",
        "source_file"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: "
                f"{column}"
            )

    result = create_windows(
        df=df,
        sensor_columns=[
            "gx",
            "gy",
            "gz"
        ],
        sampling_rate=GYROSCOPE_FS
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nTotal windows: "
        f"{result['window_id'].nunique()}"
    )

    print(
        f"Output shape: {result.shape}"
    )

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - WINDOWING")
    print("=" * 60)

    print(
        f"\nWindow length: "
        f"{WINDOW_SECONDS} seconds"
    )

    print(
        f"Overlap: "
        f"{OVERLAP * 100:.0f}%"
    )

    process_accelerometer()

    process_gyroscope()

    print("\n" + "=" * 60)
    print("WINDOWING COMPLETED ✅")
    print("=" * 60)


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":
    main()