from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "workout_summary.csv"
)

HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "workout_history.csv"
)


if not SUMMARY_FILE.exists():
    raise FileNotFoundError(
        f"Workout summary file not found: {SUMMARY_FILE}"
    )


summary_df = pd.read_csv(SUMMARY_FILE)

if summary_df.empty:
    raise ValueError(
        "Workout summary is empty."
    )


summary = summary_df.iloc[0]


summary_timestamp = datetime.fromtimestamp(
    SUMMARY_FILE.stat().st_mtime
)


session_id = summary_timestamp.strftime(
    "%Y%m%d_%H%M%S"
)


record = {
    "session_id": session_id,
    "date": summary_timestamp.strftime(
        "%Y-%m-%d"
    ),
    "time": summary_timestamp.strftime(
        "%H:%M:%S"
    ),
    "exercise": str(
        summary["exercise"]
    ),
    "detected_reps": int(
        summary["detected_reps"]
    ),
    "active_duration_sec": float(
        summary["active_duration_sec"]
    ),
    "cadence_reps_per_min": float(
        summary["cadence_reps_per_min"]
    ),
    "mean_rep_duration_sec": float(
        summary["mean_rep_duration_sec"]
    ),
    "rep_duration_cv_percent": float(
        summary["rep_duration_cv_percent"]
    ),
    "mean_waveform_correlation": float(
        summary["mean_waveform_correlation"]
    ),
    "mean_pairwise_correlation": float(
        summary["mean_pairwise_correlation"]
    ),
    "stability_score": float(
        summary["stability_score"]
    ),
    "axis_quality": float(
        summary["axis_quality"]
    ),
    "classifier_confidence_percent": float(
        summary["classifier_confidence_percent"]
    ),
}


if HISTORY_FILE.exists():
    history_df = pd.read_csv(
        HISTORY_FILE
    )
else:
    history_df = pd.DataFrame()


if (
    not history_df.empty
    and "session_id" in history_df.columns
    and session_id in history_df["session_id"].astype(str).values
):
    print("=" * 70)
    print("WORKOUT HISTORY")
    print("=" * 70)
    print()
    print(
        f"Session already exists: {session_id}"
    )
else:
    history_df = pd.concat(
        [
            history_df,
            pd.DataFrame([record]),
        ],
        ignore_index=True,
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False,
    )

    print("=" * 70)
    print("WORKOUT HISTORY")
    print("=" * 70)
    print()
    print(
        f"Session saved: {session_id}"
    )


print()
print(
    f"Exercise: {record['exercise']}"
)

print(
    f"Reps: {record['detected_reps']}"
)

print(
    f"Duration: {record['active_duration_sec']:.2f} sec"
)

print(
    f"Cadence: {record['cadence_reps_per_min']:.2f} reps/min"
)

print()
print(
    f"History file: {HISTORY_FILE}"
)

print(
    f"Total stored sessions: {len(history_df)}"
)

print()
print("WORKOUT HISTORY UPDATE COMPLETED")
