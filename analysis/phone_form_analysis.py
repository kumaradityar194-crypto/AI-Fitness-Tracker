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

ACCELEROMETER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_accelerometer_filtered.csv"
)

PER_REP_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_rep_consistency.csv"
)

SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_form_analysis.csv"
)

TARGET_POINTS = 100


def load_data():
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Segments file not found: {SEGMENTS_FILE}"
        )

    if not ACCELEROMETER_FILE.exists():
        raise FileNotFoundError(
            f"Accelerometer file not found: "
            f"{ACCELEROMETER_FILE}"
        )

    segments = pd.read_csv(
        SEGMENTS_FILE
    )

    accelerometer = pd.read_csv(
        ACCELEROMETER_FILE
    )

    required_segment_columns = [
        "rep_id",
        "exercise",
        "selected_axis",
        "start_index",
        "end_index",
        "duration_sec",
    ]

    required_accelerometer_columns = [
        "ax_filtered",
        "ay_filtered",
        "az_filtered",
    ]

    missing_segment = [
        column
        for column in required_segment_columns
        if column not in segments.columns
    ]

    missing_accelerometer = [
        column
        for column in required_accelerometer_columns
        if column not in accelerometer.columns
    ]

    if missing_segment:
        raise ValueError(
            "Missing segment columns: "
            + ", ".join(missing_segment)
        )

    if missing_accelerometer:
        raise ValueError(
            "Missing accelerometer columns: "
            + ", ".join(missing_accelerometer)
        )

    return segments, accelerometer


def normalize_waveform(signal):
    signal = np.asarray(
        signal,
        dtype=float,
    )

    if len(signal) < 2:
        return None

    signal = signal - np.mean(signal)

    scale = np.std(signal)

    if scale <= 1e-12:
        return None

    signal = signal / scale

    source_x = np.linspace(
        0.0,
        1.0,
        len(signal),
    )

    target_x = np.linspace(
        0.0,
        1.0,
        TARGET_POINTS,
    )

    normalized = np.interp(
        target_x,
        source_x,
        signal,
    )

    return normalized


def correlation_score(
    signal_a,
    signal_b,
):
    if signal_a is None or signal_b is None:
        return np.nan

    std_a = np.std(signal_a)
    std_b = np.std(signal_b)

    if std_a <= 1e-12 or std_b <= 1e-12:
        return np.nan

    correlation = np.corrcoef(
        signal_a,
        signal_b,
    )[0, 1]

    return float(correlation)


def main():
    print()
    print("=" * 75)
    print("AI FITNESS TRACKER - PHONE FORM ANALYSIS")
    print("=" * 75)

    segments, accelerometer = load_data()

    if segments.empty:
        exercise = "rest"
        selected_axis = "none"

        empty_rep_columns = [
            "rep_id",
            "duration_sec",
            "amplitude_range",
            "waveform_correlation",
            "pairwise_correlation",
        ]

        empty_rep_df = pd.DataFrame(
            columns=empty_rep_columns
        )

        empty_rep_df.to_csv(
            PER_REP_OUTPUT,
            index=False,
        )

        empty_summary = {
            "exercise": exercise,
            "repetitions_analyzed": 0,
            "mean_rep_duration_sec": 0.0,
            "rep_duration_cv_percent": 0.0,
            "mean_amplitude_range": 0.0,
            "amplitude_range_cv_percent": 0.0,
            "mean_waveform_correlation": 0.0,
            "median_waveform_correlation": 0.0,
            "mean_pairwise_correlation": 0.0,
            "median_pairwise_correlation": 0.0,
            "weakest_rep_id": 0,
            "weakest_rep_correlation": 0.0,
        }

        SUMMARY_OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            [empty_summary]
        ).to_csv(
            SUMMARY_OUTPUT,
            index=False,
        )

        print()
        print("=" * 75)
        print("MOVEMENT CONSISTENCY SUMMARY")
        print("=" * 75)

        print(
            f"Exercise                    : {exercise}"
        )

        print(
            "Repetitions analyzed        : 0"
        )

        print(
            "Mean rep duration           : 0.000 sec"
        )

        print(
            "Rep duration CV             : 0.00%"
        )

        print(
            "Mean amplitude range        : 0.0000"
        )

        print(
            "Amplitude range CV          : 0.00%"
        )

        print(
            "Mean waveform correlation   : 0.0000"
        )

        print(
            "Median waveform correlation : 0.0000"
        )

        print(
            "Mean pairwise correlation   : 0.0000"
        )

        print(
            "Median pairwise correlation : 0.0000"
        )

        print(
            "Weakest rep ID              : 0"
        )

        print(
            "Weakest rep correlation     : 0.0000"
        )

        print()
        print(
            f"Per-rep output: {PER_REP_OUTPUT}"
        )

        print(
            f"Summary output: {SUMMARY_OUTPUT}"
        )

        print()
        print("=" * 75)
        print("PHONE FORM ANALYSIS COMPLETED")
        print("=" * 75)

        return

    exercise = str(
        segments["exercise"].iloc[0]
    ).strip()

    selected_axis = str(
        segments["selected_axis"].iloc[0]
    ).strip()

    if selected_axis not in accelerometer.columns:
        raise ValueError(
            f"Selected axis not found: {selected_axis}"
        )

    print()
    print(
        f"Exercise: {exercise}"
    )

    print(
        f"Selected axis: {selected_axis}"
    )

    print(
        f"Rep segments: {len(segments)}"
    )

    waveforms = []
    amplitude_ranges = []
    rep_ids = []
    durations = []

    for _, row in segments.iterrows():
        start_index = int(
            row["start_index"]
        )

        end_index = int(
            row["end_index"]
        )

        rep_signal = (
            accelerometer[
                selected_axis
            ]
            .iloc[
                start_index:end_index + 1
            ]
            .to_numpy()
        )

        waveform = normalize_waveform(
            rep_signal
        )

        if waveform is None:
            continue

        waveforms.append(
            waveform
        )

        amplitude_ranges.append(
            float(
                np.max(rep_signal)
                - np.min(rep_signal)
            )
        )

        rep_ids.append(
            int(row["rep_id"])
        )

        durations.append(
            float(row["duration_sec"])
        )

    if len(waveforms) < 2:
        raise ValueError(
            "At least 2 valid repetitions are required."
        )

    print()
    print(
        f"Valid repetitions: {len(waveforms)}"
    )

    per_rep_results = []

    for index, waveform in enumerate(
        waveforms
    ):
        other_waveforms = (
            waveforms[:index]
            + waveforms[index + 1:]
        )

        reference = np.mean(
            np.vstack(
                other_waveforms
            ),
            axis=0,
        )

        correlation = correlation_score(
            waveform,
            reference,
        )

        per_rep_results.append(
            {
                "rep_id": rep_ids[index],
                "duration_sec": round(
                    durations[index],
                    3,
                ),
                "amplitude_range": round(
                    amplitude_ranges[index],
                    4,
                ),
                "waveform_correlation": round(
                    correlation,
                    4,
                ),
            }
        )

    per_rep_df = pd.DataFrame(
        per_rep_results
    )

    duration_series = pd.Series(
        durations,
        dtype=float,
    )

    amplitude_series = pd.Series(
        amplitude_ranges,
        dtype=float,
    )

    correlation_series = (
        per_rep_df[
            "waveform_correlation"
        ]
        .dropna()
    )

    duration_mean = (
        duration_series.mean()
    )

    duration_std = (
        duration_series.std(
            ddof=1
        )
        if len(duration_series) > 1
        else 0.0
    )

    amplitude_mean = (
        amplitude_series.mean()
    )

    amplitude_std = (
        amplitude_series.std(
            ddof=1
        )
        if len(amplitude_series) > 1
        else 0.0
    )

    duration_cv = (
        duration_std
        / duration_mean
        * 100.0
        if duration_mean > 0
        else np.nan
    )

    amplitude_cv = (
        amplitude_std
        / amplitude_mean
        * 100.0
        if amplitude_mean > 0
        else np.nan
    )

    pairwise_correlations = []

    for i in range(
        len(waveforms)
    ):
        for j in range(
            i + 1,
            len(waveforms),
        ):
            correlation = correlation_score(
                waveforms[i],
                waveforms[j],
            )

            if np.isfinite(
                correlation
            ):
                pairwise_correlations.append(
                    correlation
                )

    pairwise_series = pd.Series(
        pairwise_correlations,
        dtype=float,
    )

    mean_pairwise_correlation = (
        pairwise_series.mean()
        if not pairwise_series.empty
        else np.nan
    )

    median_pairwise_correlation = (
        pairwise_series.median()
        if not pairwise_series.empty
        else np.nan
    )

    weakest_rep = (
        per_rep_df.sort_values(
            "waveform_correlation"
        ).iloc[0]
    )

    summary = {
        "exercise": exercise,
        "repetitions_analyzed": len(
            waveforms
        ),
        "selected_axis": selected_axis,
        "mean_rep_duration_sec": round(
            duration_mean,
            3,
        ),
        "rep_duration_std_sec": round(
            duration_std,
            3,
        ),
        "rep_duration_cv_percent": round(
            duration_cv,
            2,
        ),
        "mean_amplitude_range": round(
            amplitude_mean,
            4,
        ),
        "amplitude_range_std": round(
            amplitude_std,
            4,
        ),
        "amplitude_range_cv_percent": round(
            amplitude_cv,
            2,
        ),
        "mean_waveform_correlation": round(
            correlation_series.mean(),
            4,
        ),
        "median_waveform_correlation": round(
            correlation_series.median(),
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
        "weakest_rep_id": int(
            weakest_rep["rep_id"]
        ),
        "weakest_rep_correlation": round(
            float(
                weakest_rep[
                    "waveform_correlation"
                ]
            ),
            4,
        ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    PER_REP_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_rep_df.to_csv(
        PER_REP_OUTPUT,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("=" * 75)
    print("MOVEMENT CONSISTENCY SUMMARY")
    print("=" * 75)

    print(
        f"Exercise                    : {exercise}"
    )

    print(
        f"Repetitions analyzed        : {len(waveforms)}"
    )

    print(
        f"Mean rep duration           : {duration_mean:.3f} sec"
    )

    print(
        f"Rep duration CV             : {duration_cv:.2f}%"
    )

    print(
        f"Mean amplitude range        : {amplitude_mean:.4f}"
    )

    print(
        f"Amplitude range CV          : {amplitude_cv:.2f}%"
    )

    print(
        f"Mean waveform correlation   : "
        f"{correlation_series.mean():.4f}"
    )

    print(
        f"Median waveform correlation : "
        f"{correlation_series.median():.4f}"
    )

    print(
        f"Mean pairwise correlation   : "
        f"{mean_pairwise_correlation:.4f}"
    )

    print(
        f"Median pairwise correlation : "
        f"{median_pairwise_correlation:.4f}"
    )

    print(
        f"Weakest rep ID              : "
        f"{int(weakest_rep['rep_id'])}"
    )

    print(
        f"Weakest rep correlation     : "
        f"{float(weakest_rep['waveform_correlation']):.4f}"
    )

    print()
    print(
        f"Per-rep output: {PER_REP_OUTPUT}"
    )

    print(
        f"Summary output: {SUMMARY_OUTPUT}"
    )

    print()
    print("=" * 75)
    print("PHONE FORM ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()