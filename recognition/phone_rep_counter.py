from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from recognition.rep_counter import (
    FS,
    count_repetitions_for_recording,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACCELEROMETER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_accelerometer_filtered.csv"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_prediction_diagnostics.csv"
)

STABLE_PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_stable_exercise_prediction.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_rep_counts.csv"
)


def load_predictions():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_FILE}"
        )

    prediction_df = pd.read_csv(
        PREDICTION_FILE
    )

    required_columns = [
        "predicted_exercise",
        "confidence_percent",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in prediction_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(missing_columns)
        )

    prediction_df = (
        prediction_df
        .dropna(subset=required_columns)
        .reset_index(drop=True)
    )

    if prediction_df.empty:
        raise ValueError(
            "Prediction file contains no valid rows."
        )

    return prediction_df


def select_exercise(prediction_df):
    exercise_counts = (
        prediction_df["predicted_exercise"]
        .value_counts()
    )

    selected_exercise = (
        exercise_counts.index[0]
    )

    return (
        selected_exercise,
        exercise_counts,
    )


def load_stable_prediction():
    if not STABLE_PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Stable prediction file not found: "
            f"{STABLE_PREDICTION_FILE}"
        )

    stable_df = pd.read_csv(
        STABLE_PREDICTION_FILE
    )

    required_columns = [
        "final_exercise",
        "average_classifier_confidence_percent",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in stable_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing stable prediction columns: "
            + ", ".join(missing_columns)
        )

    if stable_df.empty:
        raise ValueError(
            "Stable prediction file contains no rows."
        )

    selected_exercise = str(
        stable_df.iloc[0]["final_exercise"]
    )

    average_confidence = float(
        stable_df.iloc[0][
            "average_classifier_confidence_percent"
        ]
    )

    if not selected_exercise:
        raise ValueError(
            "Stable prediction contains an empty exercise."
        )

    return (
        selected_exercise,
        average_confidence,
    )


def load_phone_accelerometer():
    if not ACCELEROMETER_FILE.exists():
        raise FileNotFoundError(
            f"Phone accelerometer file not found: "
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
            "Missing accelerometer columns: "
            + ", ".join(missing_columns)
        )

    df = (
        df[required_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
        .reset_index(drop=True)
    )

    minimum_samples = int(FS * 5)

    if len(df) < minimum_samples:
        raise ValueError(
            "Phone recording is too short for rep counting."
        )

    return df



def detect_stationary_session(
    df,
    fs,
    window_sec=0.5,
    std_threshold=0.03,
    range_threshold=0.12,
    stationary_ratio_threshold=0.80,
):
    axis_columns = [
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
    ]

    numeric_df = (
        df[axis_columns]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )

    if len(numeric_df) < max(20, int(fs * 2)):
        return False, {
            "combined_std": float("inf"),
            "maximum_range": float("inf"),
            "stationary_ratio": 0.0,
        }

    axis_std = numeric_df.std()
    axis_range = numeric_df.max() - numeric_df.min()

    combined_std = float(
        np.linalg.norm(axis_std.to_numpy(dtype=float))
    )

    maximum_range = float(
        axis_range.max()
    )

    window_samples = max(
        5,
        int(fs * window_sec),
    )

    window_flags = []

    for start in range(
        0,
        len(numeric_df) - window_samples + 1,
        window_samples,
    ):
        window = numeric_df.iloc[
            start:start + window_samples
        ]

        local_std = window.std()
        local_range = window.max() - window.min()

        local_combined_std = float(
            np.linalg.norm(
                local_std.to_numpy(dtype=float)
            )
        )

        local_maximum_range = float(
            local_range.max()
        )

        window_flags.append(
            (
                local_combined_std <= std_threshold
                and local_maximum_range <= range_threshold
            )
        )

    if window_flags:
        stationary_ratio = float(
            np.mean(window_flags)
        )
    else:
        stationary_ratio = 0.0

    is_stationary = (
        combined_std <= 0.12
        and stationary_ratio >= 0.80
    )

    return is_stationary, {
        "combined_std": combined_std,
        "maximum_range": maximum_range,
        "stationary_ratio": stationary_ratio,
    }


def trim_inactive_tail(
    df,
    fs,
    window_sec=0.5,
    required_inactive_windows=3,
    std_threshold=0.03,
):
    axis_columns = [
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
    ]

    available_axes = [
        column
        for column in axis_columns
        if column in df.columns
    ]

    if not available_axes:
        return df.copy()

    window_samples = max(
        1,
        int(fs * window_sec),
    )

    total_windows = (
        len(df) // window_samples
    )

    window_records = []

    for window_index in range(
        total_windows
    ):
        start = (
            window_index
            * window_samples
        )

        end = (
            start
            + window_samples
        )

        window = df.iloc[start:end]

        axis_std = [
            window[column].std()
            for column in available_axes
        ]

        activity_std = max(
            axis_std
        )

        window_records.append(
            {
                "start": start,
                "end": end,
                "activity_std": activity_std,
            }
        )

    inactive_run = 0
    cutoff_index = None

    for record in reversed(
        window_records
    ):
        if (
            record["activity_std"]
            < std_threshold
        ):
            inactive_run += 1

            if (
                inactive_run
                >= required_inactive_windows
            ):
                cutoff_index = (
                    record["start"]
                )
        else:
            break

    if cutoff_index is None:
        return df.copy()

    minimum_keep_samples = int(
        fs * 5
    )

    if cutoff_index < minimum_keep_samples:
        return df.copy()

    return (
        df.iloc[:cutoff_index]
        .copy()
        .reset_index(drop=True)
    )


def save_result(
    result,
    selected_exercise,
    original_samples,
    final_samples,
):
    result_row = {
        "exercise": selected_exercise,
        "selected_axis": result.get(
            "selected_axis"
        ),
        "estimated_period_sec": result.get(
            "estimated_period_sec"
        ),
        "estimated_frequency_hz": result.get(
            "estimated_frequency_hz"
        ),
        "period_score": result.get(
            "period_score"
        ),
        "stability_score": result.get(
            "stability_score"
        ),
        "axis_quality": result.get(
            "axis_quality"
        ),
        "detected_reps": result.get(
            "detected_reps"
        ),
        "counting_method": result.get(
            "counting_method"
        ),
        "candidate_counts": result.get(
            "candidate_counts"
        ),
        "candidate_scores": result.get(
            "candidate_scores"
        ),
        "original_samples": original_samples,
        "final_samples": final_samples,
        "trimmed_samples": (
            original_samples
            - final_samples
        ),
        "duration_sec": (
            final_samples / FS
        ),
    }

    result_df = pd.DataFrame(
        [result_row]
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return result_df


def main():
    print()
    print("=" * 70)
    print("AI FITNESS TRACKER - PHONE REP COUNTER")
    print("=" * 70)

    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Prediction file: "
        f"{PREDICTION_FILE}"
    )

    print(
        f"Accelerometer file: "
        f"{ACCELEROMETER_FILE}"
    )

    print()
    print(
        "Loading exercise predictions..."
    )

    try:
        (
            selected_exercise,
            average_confidence,
        ) = load_stable_prediction()

        prediction_source = "stable"

        print()
        print(
            "Stable prediction file:"
        )

        print(
            f"{STABLE_PREDICTION_FILE}"
        )

        print()
        print(
            f"Stable exercise: "
            f"{selected_exercise}"
        )

    except FileNotFoundError:
        prediction_df = load_predictions()

        (
            selected_exercise,
            exercise_counts,
        ) = select_exercise(
            prediction_df
        )

        average_confidence = (
            prediction_df[
                "confidence_percent"
            ].mean()
        )

        prediction_source = "raw"

        print()
        print(
            "Stable prediction not found."
        )

        print(
            "Using raw prediction fallback."
        )

        print()
        print(
            "Prediction distribution:"
        )

        print(
            exercise_counts.to_string()
        )

    print()
    print(
        f"Prediction source: "
        f"{prediction_source}"
    )

    print(
        f"Selected exercise: "
        f"{selected_exercise}"
    )

    print(
        f"Average classifier confidence: "
        f"{average_confidence:.2f}%"
    )

    print()
    print(
        "Loading phone accelerometer data..."
    )

    df = load_phone_accelerometer()

    original_samples = len(df)

    print(
        f"Loaded samples: "
        f"{original_samples:,}"
    )

    original_duration = (
        original_samples / FS
    )

    print(
        f"Original duration: "
        f"{original_duration:.2f} seconds"
    )

    print()
    print(
        "Detecting inactive tail..."
    )

    df = trim_inactive_tail(
        df,
        FS,
    )

    final_samples = len(df)

    trimmed_samples = (
        original_samples
        - final_samples
    )

    if trimmed_samples > 0:
        print(
            f"Inactive tail removed: "
            f"{trimmed_samples:,} samples"
        )

        print(
            f"Active duration: "
            f"{final_samples / FS:.2f} seconds"
        )
    else:
        print(
            "No inactive tail detected."
        )


    stationary, motion_metrics = detect_stationary_session(
        df,
        FS,
    )

    print()
    print("Stationary motion check:")
    print(
        f"Combined accelerometer standard deviation: "
        f"{motion_metrics['combined_std']:.6f} g"
    )
    print(
        f"Maximum accelerometer range: "
        f"{motion_metrics['maximum_range']:.6f} g"
    )
    print(
        f"Stationary windows ratio: "
        f"{motion_metrics['stationary_ratio']:.2%}"
    )

    if stationary:
        print()
        print("Stationary session detected.")
        print(
            "Overriding exercise prediction to rest."
        )
        print(
            "Skipping rep counting because no significant motion was detected."
        )

        stationary_result = {
            "selected_axis": "none",
            "estimated_period_sec": 0.0,
            "estimated_frequency_hz": 0.0,
            "period_score": 0.0,
            "stability_score": 0.0,
            "axis_quality": 0.0,
            "detected_reps": 0,
            "counting_method": "stationary_gate",
            "candidate_counts": "0",
            "candidate_scores": "0.0000",
        }

        save_result(
            stationary_result,
            "rest",
            original_samples,
            final_samples,
        )

        print()
        print("Exercise            : rest")
        print("Detected reps       : 0")
        print("Counting method     : stationary_gate")
        print()
        print(f"Saved: {OUTPUT_FILE}")
        print()
        print("=" * 70)
        print("PHONE REP COUNTING COMPLETED")
        print("=" * 70)
        return

    df["exercise"] = (
        selected_exercise
    )

    df["intensity"] = "unknown"

    print()
    print(
        "Running V8 rep counter..."
    )

    result = count_repetitions_for_recording(
        df,
        FS,
    )

    print()
    print("=" * 70)
    print("REP COUNT RESULT")
    print("=" * 70)

    print(
        f"Exercise            : "
        f"{selected_exercise}"
    )

    print(
        f"Selected axis       : "
        f"{result['selected_axis']}"
    )

    print(
        f"Estimated period    : "
        f"{result['estimated_period_sec']:.2f}"
    )

    print(
        f"Estimated frequency : "
        f"{result['estimated_frequency_hz']:.3f}"
    )

    print(
        f"Period score        : "
        f"{result['period_score']:.4f}"
    )

    print(
        f"Stability score     : "
        f"{result['stability_score']:.4f}"
    )

    print(
        f"Axis quality        : "
        f"{result['axis_quality']:.4f}"
    )

    print(
        f"Counting method     : "
        f"{result['counting_method']}"
    )

    print(
        f"Detected reps       : "
        f"{result['detected_reps']}"
    )

    print(
        f"Candidate counts    : "
        f"{result['candidate_counts']}"
    )

    print(
        f"Candidate scores    : "
        f"{result['candidate_scores']}"
    )

    save_result(
        result,
        selected_exercise,
        original_samples,
        final_samples,
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print(
        "PHONE REP COUNTING COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()


