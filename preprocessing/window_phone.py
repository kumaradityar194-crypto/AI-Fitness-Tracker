from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


ACC_INPUT = (
    PROCESSED_DIR /
    "phone_accelerometer_filtered.csv"
)

GYRO_INPUT = (
    PROCESSED_DIR /
    "phone_gyroscope_filtered.csv"
)

ACC_OUTPUT = (
    PROCESSED_DIR /
    "phone_accelerometer_windowed.csv"
)

GYRO_OUTPUT = (
    PROCESSED_DIR /
    "phone_gyroscope_windowed.csv"
)


# ============================================================
# 2. WINDOW SETTINGS
# ============================================================

WINDOW_SECONDS = 4

OVERLAP = 0.50

# Existing model pipeline settings
ACC_FS = 12.5
GYRO_FS = 25.0


# ============================================================
# 3. PREPARE TIMESTAMP
# ============================================================

def prepare_timestamp(df):
    """
    Ensure the dataframe contains timestamp_ms.

    Resampled phone files contain a datetime column called
    'time', so convert it back to epoch milliseconds.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Case 1: timestamp_ms already exists
    # --------------------------------------------------------

    if "timestamp_ms" in df.columns:

        df["timestamp_ms"] = pd.to_numeric(
            df["timestamp_ms"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Case 2: resampled file contains 'time'
    # --------------------------------------------------------

    elif "time" in df.columns:

        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce"
        )

        if df["time"].isna().all():

            raise ValueError(
                "Column 'time' exists but contains "
                "no valid timestamps."
            )

        df["timestamp_ms"] = (
            df["time"]
            .astype("int64")
            // 1_000_000
        )

    else:

        raise ValueError(
            "Neither 'timestamp_ms' nor 'time' "
            "column was found."
        )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["timestamp_ms"]
    ).copy()

    df["timestamp_ms"] = (
        df["timestamp_ms"]
        .astype("int64")
    )

    return df


# ============================================================
# 4. CREATE WINDOWS
# ============================================================

def create_windows(
    df,
    sampling_rate
):
    """
    Divide sensor data into fixed-size overlapping windows.

    Important:
    Each session is processed separately so that samples
    from different recording sessions are never mixed.
    """

    df = prepare_timestamp(df)

    # --------------------------------------------------------
    # Window size
    # --------------------------------------------------------

    samples_per_window = int(
        WINDOW_SECONDS * sampling_rate
    )

    # --------------------------------------------------------
    # Step size
    #
    # 50% overlap:
    # step = 50% of window
    # --------------------------------------------------------

    step_size = int(
        samples_per_window *
        (1 - OVERLAP)
    )

    print(
        f"\nSampling rate: {sampling_rate} Hz"
    )

    print(
        f"Samples per window: "
        f"{samples_per_window}"
    )

    print(
        f"Step size: {step_size}"
    )

    # --------------------------------------------------------
    # Check session_id
    # --------------------------------------------------------

    if "session_id" not in df.columns:

        raise ValueError(
            "session_id column not found."
        )

    windows = []

    window_id = 0

    # ========================================================
    # Process every session separately
    # ========================================================

    for session_id, session in df.groupby(
        "session_id",
        sort=False
    ):

        session = (
            session
            .sort_values("timestamp_ms")
            .reset_index(drop=True)
        )

        total_samples = len(session)

        print(
            f"\nSession {session_id}: "
            f"{total_samples} samples"
        )

        # ----------------------------------------------------
        # Not enough data for even one window
        # ----------------------------------------------------

        if total_samples < samples_per_window:

            print(
                f"Session {session_id} skipped: "
                f"not enough samples."
            )

            continue

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

            window = session.iloc[
                start:end
            ].copy()

            # ------------------------------------------------
            # Window metadata
            # ------------------------------------------------

            window["window_id"] = window_id

            window["window_sample"] = range(
                len(window)
            )

            window_start_timestamp = (
                window["timestamp_ms"]
                .iloc[0]
            )

            window_end_timestamp = (
                window["timestamp_ms"]
                .iloc[-1]
            )

            session_start_timestamp = (
                session["timestamp_ms"]
                .iloc[0]
            )

            window["window_start_s"] = (
                window_start_timestamp
                - session_start_timestamp
            ) / 1000.0

            window["window_end_s"] = (
                window_end_timestamp
                - session_start_timestamp
            ) / 1000.0

            windows.append(window)

            window_id += 1

    # ========================================================
    # No windows
    # ========================================================

    if not windows:

        raise ValueError(
            "No windows could be created. "
            "Check recording length and sampling rate."
        )

    # ========================================================
    # Combine
    # ========================================================

    result = pd.concat(
        windows,
        ignore_index=True
    )

    return result


# ============================================================
# 5. ACCELEROMETER WINDOWING
# ============================================================

def process_accelerometer():

    print("\n" + "=" * 60)
    print("PHONE ACCELEROMETER WINDOWING")
    print("=" * 60)

    if not ACC_INPUT.exists():

        raise FileNotFoundError(
            f"Accelerometer file not found:\n"
            f"{ACC_INPUT}"
        )

    df = pd.read_csv(
        ACC_INPUT
    )

    print(
        f"\nInput shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "ax",
        "ay",
        "az",
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
        "session_id"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Accelerometer missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    result = create_windows(
        df=df,
        sampling_rate=ACC_FS
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        ACC_OUTPUT,
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
        f"\nSaved:\n{ACC_OUTPUT}"
    )


# ============================================================
# 6. GYROSCOPE WINDOWING
# ============================================================

def process_gyroscope():

    print("\n" + "=" * 60)
    print("PHONE GYROSCOPE WINDOWING")
    print("=" * 60)

    if not GYRO_INPUT.exists():

        raise FileNotFoundError(
            f"Gyroscope file not found:\n"
            f"{GYRO_INPUT}"
        )

    df = pd.read_csv(
        GYRO_INPUT
    )

    print(
        f"\nInput shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "gx",
        "gy",
        "gz",
        "gx_filtered",
        "gy_filtered",
        "gz_filtered",
        "session_id"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Gyroscope missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    result = create_windows(
        df=df,
        sampling_rate=GYRO_FS
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        GYRO_OUTPUT,
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
        f"\nSaved:\n{GYRO_OUTPUT}"
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER - PHONE WINDOWING")
    print("=" * 60)

    print(
        f"\nWindow size: "
        f"{WINDOW_SECONDS} seconds"
    )

    print(
        f"Overlap: "
        f"{OVERLAP * 100:.1f}%"
    )

    print(
        f"Accelerometer: "
        f"{ACC_FS} Hz"
    )

    print(
        f"Gyroscope: "
        f"{GYRO_FS} Hz"
    )

    process_accelerometer()

    process_gyroscope()

    print("\n" + "=" * 60)
    print("PHONE WINDOWING COMPLETED")
    print("=" * 60)


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":
    main()