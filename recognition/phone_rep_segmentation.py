from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from recognition.rep_counter import (
    FS,
    get_exercise_config,
    prepare_signal,
    fundamental_filter,
    safe_savgol,
    detect_peaks,
    clean_peaks,
    remove_boundary_peaks,
)

from recognition.phone_rep_counter import (
    trim_inactive_tail,
)


ACCELEROMETER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_accelerometer_filtered.csv"
)

REP_RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_rep_counts.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_reps"
)

SEGMENT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_rep_segments.csv"
)


def load_rep_result():
    if not REP_RESULT_FILE.exists():
        raise FileNotFoundError(
            f"Rep result file not found: {REP_RESULT_FILE}"
        )

    result_df = pd.read_csv(
        REP_RESULT_FILE
    )

    if result_df.empty:
        raise ValueError(
            "Rep result file contains no rows."
        )

    required_columns = [
        "exercise",
        "selected_axis",
        "estimated_period_sec",
        "detected_reps",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    row = result_df.iloc[0]

    exercise = str(
        row["exercise"]
    ).lower().strip()

    selected_axis = str(
        row["selected_axis"]
    ).strip()

    period = float(
        row["estimated_period_sec"]
    )

    detected_reps = int(
        row["detected_reps"]
    )

    if selected_axis == "none" and detected_reps == 0:
        return (
            exercise,
            selected_axis,
            period,
            detected_reps,
        )

    if selected_axis == "none":
        raise ValueError(
            "No valid axis was selected by the rep counter."
        )

    if not np.isfinite(period) or period <= 0:
        raise ValueError(
            "Invalid estimated period."
        )

    if detected_reps <= 0:
        raise ValueError(
            "No repetitions were detected."
        )

    return (
        exercise,
        selected_axis,
        period,
        detected_reps,
    )


def load_accelerometer():
    if not ACCELEROMETER_FILE.exists():
        raise FileNotFoundError(
            f"Accelerometer file not found: "
            f"{ACCELEROMETER_FILE}"
        )

    df = pd.read_csv(
        ACCELEROMETER_FILE
    )

    required_columns = [
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing accelerometer columns: "
            f"{missing_columns}"
        )

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna(
            subset=required_columns
        )
        .reset_index(
            drop=True
        )
    )

    if len(df) < int(FS * 5):
        raise ValueError(
            "Accelerometer recording is too short."
        )

    return df


def detect_rep_centers(
    df,
    axis,
    exercise,
    period,
):
    config = get_exercise_config(
        exercise,
        "unknown",
    )

    raw_signal = pd.to_numeric(
        df[axis],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    broad_signal = prepare_signal(
        raw_signal,
        FS,
        config,
    )

    if len(broad_signal) < int(FS * 5):
        raise ValueError(
            "Signal is too short for rep segmentation."
        )

    signal = fundamental_filter(
        broad_signal,
        FS,
        period,
    )

    signal = safe_savgol(
        signal,
        FS,
        max(
            0.40,
            config["smooth_sec"] * 0.55,
        ),
    )

    peaks, direction, detector_score = detect_peaks(
        signal,
        FS,
        period,
        config,
        1.00,
        1.00,
    )

    peaks = clean_peaks(
        peaks,
        FS,
        period,
        config,
    )

    peaks = remove_boundary_peaks(
        peaks,
        len(signal),
        FS,
        period,
    )

    return (
        signal,
        peaks,
        direction,
        detector_score,
    )


def create_segments(
    df,
    signal,
    peaks,
    period,
):
    if len(peaks) == 0:
        raise ValueError(
            "No rep centers were detected."
        )

    half_period_samples = max(
        1,
        int(
            round(
                period
                * FS
                / 2.0
            )
        ),
    )

    boundaries = []

    for index in range(len(peaks) + 1):
        if index == 0:
            boundary = max(
                0,
                peaks[0]
                - half_period_samples,
            )

        elif index == len(peaks):
            boundary = min(
                len(df),
                peaks[-1]
                + half_period_samples,
            )

        else:
            midpoint = int(
                round(
                    (
                        peaks[index - 1]
                        + peaks[index]
                    )
                    / 2.0
                )
            )

            boundary = midpoint

        boundaries.append(
            boundary
        )

    segments = []

    for rep_index, peak in enumerate(
        peaks,
        start=1,
    ):
        start_idx = boundaries[
            rep_index - 1
        ]

        end_idx = boundaries[
            rep_index
        ]

        if end_idx <= start_idx:
            continue

        segment = df.iloc[
            start_idx:end_idx
        ].copy()

        segment["rep_id"] = rep_index
        segment["relative_time_s"] = (
            np.arange(
                len(segment)
            )
            / FS
        )

        segment["rep_center_index"] = (
            peak
        )

        segment["rep_center_time_s"] = (
            peak / FS
        )

        segments.append(
            {
                "rep_id": rep_index,
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "center_index": peak,
                "start_time_s": round(
                    start_idx / FS,
                    3,
                ),
                "center_time_s": round(
                    peak / FS,
                    3,
                ),
                "end_time_s": round(
                    (end_idx - 1) / FS,
                    3,
                ),
                "duration_sec": round(
                    (end_idx - start_idx) / FS,
                    3,
                ),
                "sample_count": len(segment),
            }
        )

        segment_file = (
            OUTPUT_DIR
            / f"rep_{rep_index:02d}.csv"
        )

        segment.to_csv(
            segment_file,
            index=False,
        )

    return segments


def main():
    print()
    print("=" * 70)
    print("AI FITNESS TRACKER - PHONE REP SEGMENTATION")
    print("=" * 70)

    print()
    print("Loading rep counter result...")

    (
        exercise,
        selected_axis,
        period,
        detected_reps,
    ) = load_rep_result()

    print(
        f"Exercise: {exercise}"
    )

    print(
        f"Selected axis: {selected_axis}"
    )

    print(
        f"Estimated period: {period:.3f} seconds"
    )

    print(
        f"Detected reps: {detected_reps}"
    )

    print()
    if selected_axis == "none" and detected_reps == 0:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        for segment_file in OUTPUT_DIR.glob("rep_*.csv"):
            segment_file.unlink()

        empty_columns = [
            "rep_id",
            "exercise",
            "selected_axis",
            "estimated_period_sec",
            "start_index",
            "end_index",
            "center_index",
            "start_time_s",
            "center_time_s",
            "end_time_s",
            "duration_sec",
            "sample_count",
        ]

        pd.DataFrame(
            columns=empty_columns
        ).to_csv(
            SEGMENT_SUMMARY_FILE,
            index=False,
        )

        print()
        print("No repetitions detected. Skipping rep segmentation.")
        print(
            f"Saved empty segment summary: "
            f"{SEGMENT_SUMMARY_FILE}"
        )

        print()
        print("=" * 70)
        print("PHONE REP SEGMENTATION COMPLETED")
        print("=" * 70)
        return

    print("Loading accelerometer data...")

    df = load_accelerometer()

    print(
        f"Samples loaded: {len(df):,}"
    )

    original_samples = len(df)

    print()
    print("Detecting inactive tail...")

    df = trim_inactive_tail(
        df,
        FS,
    )

    print(
        f"Original samples: {original_samples}"
    )

    print(
        f"Active samples: {len(df)}"
    )

    print(
        f"Active duration: {len(df) / FS:.2f} seconds"
    )

    print()
    print("Detecting rep centers...")

    (
        signal,
        peaks,
        direction,
        detector_score,
    ) = detect_rep_centers(
        df,
        selected_axis,
        exercise,
        period,
    )

    print(
        f"Detected peak centers: {len(peaks)}"
    )

    print(
        f"Detection direction: {direction}"
    )

    print(
        f"Detector score: {detector_score:.4f}"
    )

    if len(peaks) != detected_reps:
        raise ValueError(
            f"Segmentation detected {len(peaks)} centers, "
            f"but rep counter reported {detected_reps} reps."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Creating rep segments...")

    segments = create_segments(
        df,
        signal,
        peaks,
        period,
    )

    summary_df = pd.DataFrame(
        segments
    )

    summary_df.insert(
        1,
        "exercise",
        exercise,
    )

    summary_df.insert(
        2,
        "selected_axis",
        selected_axis,
    )

    summary_df.insert(
        3,
        "estimated_period_sec",
        period,
    )

    summary_df.to_csv(
        SEGMENT_SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("REP SEGMENTATION RESULT")
    print("=" * 70)

    print(
        f"Exercise          : {exercise}"
    )

    print(
        f"Detected reps     : {len(peaks)}"
    )

    print(
        f"Selected axis     : {selected_axis}"
    )

    print(
        f"Detection direction: {direction}"
    )

    print()
    print("Rep segments:")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Segment directory: {OUTPUT_DIR}"
    )

    print(
        f"Summary file: {SEGMENT_SUMMARY_FILE}"
    )

    print()
    print("=" * 70)
    print("PHONE REP SEGMENTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()





