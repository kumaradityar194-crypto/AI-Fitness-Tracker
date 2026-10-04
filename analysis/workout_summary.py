from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_rep_segments.csv"
)

REP_RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_rep_counts.csv"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_prediction_diagnostics.csv"
)

FORM_ANALYSIS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_form_analysis.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "workout_summary.csv"
)


def load_file(path, allow_empty=False):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty and not allow_empty:
        raise ValueError(
            f"File contains no data: {path}"
        )

    return df


def main():
    print()
    print("=" * 80)
    print("AI FITNESS TRACKER - FINAL WORKOUT SUMMARY")
    print("=" * 80)

    print()
    print("Loading workout data...")

    segments_df = load_file(
        SEGMENTS_FILE,
        allow_empty=True,
    )

    rep_result_df = load_file(
        REP_RESULT_FILE
    )

    prediction_df = load_file(
        PREDICTION_FILE
    )

    form_analysis_df = load_file(
        FORM_ANALYSIS_FILE
    )

    rep_result = rep_result_df.iloc[0]
    form_result = form_analysis_df.iloc[0]

    exercise = str(
        rep_result["exercise"]
    ).strip()

    detected_reps = int(
        rep_result["detected_reps"]
    )

    selected_axis = str(
        rep_result["selected_axis"]
    ).strip()

    estimated_period = float(
        rep_result["estimated_period_sec"]
    )

    estimated_frequency = float(
        rep_result["estimated_frequency_hz"]
    )

    period_score = float(
        rep_result["period_score"]
    )

    stability_score = float(
        rep_result["stability_score"]
    )

    axis_quality = float(
        rep_result["axis_quality"]
    )

    counting_method = str(
        rep_result["counting_method"]
    ).strip()

    original_samples = int(
        rep_result["original_samples"]
    )

    final_samples = int(
        rep_result["final_samples"]
    )

    trimmed_samples = int(
        rep_result["trimmed_samples"]
    )

    active_duration = float(
        rep_result["duration_sec"]
    )

    classifier_confidence = float(
        prediction_df[
            "confidence_percent"
        ].mean()
    )

    exercise_window_percentage = float(
        prediction_df[
            "predicted_exercise"
        ]
        .astype(str)
        .str.lower()
        .eq(exercise.lower())
        .mean()
        * 100.0
    )

    cadence = (
        60.0 / estimated_period
        if estimated_period > 0
        else 0.0
    )

    if segments_df.empty:
        mean_rep_duration = 0.0
        median_rep_duration = 0.0
        rep_duration_std = 0.0
        rep_duration_cv = 0.0
    else:
        mean_rep_duration = float(
            segments_df[
                "duration_sec"
            ]
            .mean()
        )

        median_rep_duration = float(
            segments_df[
                "duration_sec"
            ]
            .median()
        )

        rep_duration_std = float(
            segments_df[
                "duration_sec"
            ]
            .std(ddof=1)
            if len(segments_df) > 1
            else 0.0
        )

        rep_duration_cv = (
            rep_duration_std
            / mean_rep_duration
            * 100.0
            if mean_rep_duration > 0
            else 0.0
        )

    mean_amplitude_range = float(
        form_result[
            "mean_amplitude_range"
        ]
    )

    amplitude_range_cv = float(
        form_result[
            "amplitude_range_cv_percent"
        ]
    )

    mean_waveform_correlation = float(
        form_result[
            "mean_waveform_correlation"
        ]
    )

    median_waveform_correlation = float(
        form_result[
            "median_waveform_correlation"
        ]
    )

    mean_pairwise_correlation = float(
        form_result[
            "mean_pairwise_correlation"
        ]
    )

    median_pairwise_correlation = float(
        form_result[
            "median_pairwise_correlation"
        ]
    )

    weakest_rep_id = int(
        form_result[
            "weakest_rep_id"
        ]
    )

    weakest_rep_correlation = float(
        form_result[
            "weakest_rep_correlation"
        ]
    )

    summary = {
        "exercise": exercise,
        "detected_reps": detected_reps,
        "active_duration_sec": round(
            active_duration,
            2,
        ),
        "original_samples": original_samples,
        "active_samples": final_samples,
        "trimmed_samples": trimmed_samples,
        "selected_axis": selected_axis,
        "estimated_period_sec": round(
            estimated_period,
            3,
        ),
        "estimated_frequency_hz": round(
            estimated_frequency,
            4,
        ),
        "cadence_reps_per_min": round(
            cadence,
            2,
        ),
        "period_score": round(
            period_score,
            4,
        ),
        "stability_score": round(
            stability_score,
            4,
        ),
        "axis_quality": round(
            axis_quality,
            4,
        ),
        "counting_method": counting_method,
        "classifier_confidence_percent": round(
            classifier_confidence,
            2,
        ),
        "exercise_window_percentage": round(
            exercise_window_percentage,
            2,
        ),
        "mean_rep_duration_sec": round(
            mean_rep_duration,
            3,
        ),
        "median_rep_duration_sec": round(
            median_rep_duration,
            3,
        ),
        "rep_duration_std_sec": round(
            rep_duration_std,
            3,
        ),
        "rep_duration_cv_percent": round(
            rep_duration_cv,
            2,
        ),
        "mean_amplitude_range": round(
            mean_amplitude_range,
            4,
        ),
        "amplitude_range_cv_percent": round(
            amplitude_range_cv,
            2,
        ),
        "mean_waveform_correlation": round(
            mean_waveform_correlation,
            4,
        ),
        "median_waveform_correlation": round(
            median_waveform_correlation,
            4,
        ),
        "mean_pairwise_correlation": round(
            mean_pairwise_correlation,
            4,
        ),
        "median_pairwise_correlation": round(
            median_pairwise_correlation,
            4,
        ),
        "weakest_rep_id": weakest_rep_id,
        "weakest_rep_correlation": round(
            weakest_rep_correlation,
            4,
        ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL WORKOUT SUMMARY")
    print("=" * 80)

    print(
        f"Exercise                         : "
        f"{exercise}"
    )

    print(
        f"Detected reps                    : "
        f"{detected_reps}"
    )

    print(
        f"Active duration                  : "
        f"{active_duration:.2f} sec"
    )

    print(
        f"Estimated rep period             : "
        f"{estimated_period:.3f} sec"
    )

    print(
        f"Cadence                          : "
        f"{cadence:.2f} reps/min"
    )

    print(
        f"Mean rep duration                : "
        f"{mean_rep_duration:.3f} sec"
    )

    print(
        f"Rep duration variation            : "
        f"{rep_duration_cv:.2f}%"
    )

    print(
        f"Amplitude range variation         : "
        f"{amplitude_range_cv:.2f}%"
    )

    print(
        f"Mean waveform correlation         : "
        f"{mean_waveform_correlation:.4f}"
    )

    print(
        f"Median waveform correlation       : "
        f"{median_waveform_correlation:.4f}"
    )

    print(
        f"Mean pairwise correlation         : "
        f"{mean_pairwise_correlation:.4f}"
    )

    print(
        f"Median pairwise correlation       : "
        f"{median_pairwise_correlation:.4f}"
    )

    print(
        f"Weakest rep ID                   : "
        f"{weakest_rep_id}"
    )

    print(
        f"Weakest rep correlation           : "
        f"{weakest_rep_correlation:.4f}"
    )

    print(
        f"Stability score                   : "
        f"{stability_score:.4f}"
    )

    print(
        f"Axis quality                      : "
        f"{axis_quality:.4f}"
    )

    print(
        f"Classifier confidence             : "
        f"{classifier_confidence:.2f}%"
    )

    print(
        f"Exercise windows                  : "
        f"{exercise_window_percentage:.2f}%"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print("=" * 80)
    print("FINAL WORKOUT SUMMARY COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()