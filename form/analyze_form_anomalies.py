from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "form_anomaly_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "form_analysis"
)

PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. LOAD DATA
# =========================================================

print("=" * 80)
print("AI FITNESS TRACKER")
print("FORM ANOMALY ANALYSIS")
print("=" * 80)

print("\nLoading anomaly results...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nFile not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")

# Required columns
required_columns = [
    "participant",
    "exercise",
    "intensity",
    "form_anomaly_prediction",
    "form_anomaly_score",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns:\n{missing_columns}"
    )


# =========================================================
# 3. BASIC CLEANING
# =========================================================

df = df.copy()

df["form_anomaly_score"] = pd.to_numeric(
    df["form_anomaly_score"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "participant",
        "exercise",
        "intensity",
        "form_anomaly_score",
    ]
)

print(
    f"Rows after cleaning: {len(df)}"
)


# =========================================================
# 4. CREATE BINARY ANOMALY COLUMN
# =========================================================

df["is_unusual"] = (
    df["form_anomaly_prediction"]
    .astype(str)
    .str.lower()
    .eq("unusual")
    .astype(int)
)


# =========================================================
# 5. OVERALL SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

total_samples = len(df)

normal_samples = int(
    (df["is_unusual"] == 0).sum()
)

unusual_samples = int(
    (df["is_unusual"] == 1).sum()
)

overall_unusual_percentage = (
    unusual_samples
    / total_samples
    * 100
)

print(f"Total samples       : {total_samples}")
print(f"Normal samples      : {normal_samples}")
print(f"Unusual samples     : {unusual_samples}")
print(
    f"Unusual percentage  : "
    f"{overall_unusual_percentage:.2f}%"
)

print(
    f"Mean anomaly score  : "
    f"{df['form_anomaly_score'].mean():.6f}"
)

print(
    f"Median anomaly score: "
    f"{df['form_anomaly_score'].median():.6f}"
)

print(
    f"Minimum score       : "
    f"{df['form_anomaly_score'].min():.6f}"
)

print(
    f"Maximum score       : "
    f"{df['form_anomaly_score'].max():.6f}"
)


# =========================================================
# 6. EXERCISE-WISE ANALYSIS
# =========================================================

print("\n" + "=" * 80)
print("EXERCISE-WISE ANALYSIS")
print("=" * 80)

exercise_summary = (
    df
    .groupby("exercise")
    .agg(
        total_samples=("is_unusual", "count"),
        unusual_samples=("is_unusual", "sum"),
        mean_score=("form_anomaly_score", "mean"),
        median_score=("form_anomaly_score", "median"),
        min_score=("form_anomaly_score", "min"),
        max_score=("form_anomaly_score", "max"),
    )
    .reset_index()
)

exercise_summary["unusual_percentage"] = (
    exercise_summary["unusual_samples"]
    / exercise_summary["total_samples"]
    * 100
)

exercise_summary = exercise_summary.sort_values(
    by="mean_score",
    ascending=False
)

print(
    exercise_summary.to_string(index=False)
)

exercise_summary.to_csv(
    OUTPUT_DIR / "exercise_anomaly_summary.csv",
    index=False
)


# =========================================================
# 7. PARTICIPANT-WISE ANALYSIS
# =========================================================

print("\n" + "=" * 80)
print("PARTICIPANT-WISE ANALYSIS")
print("=" * 80)

participant_summary = (
    df
    .groupby("participant")
    .agg(
        total_samples=("is_unusual", "count"),
        unusual_samples=("is_unusual", "sum"),
        mean_score=("form_anomaly_score", "mean"),
        median_score=("form_anomaly_score", "median"),
        max_score=("form_anomaly_score", "max"),
    )
    .reset_index()
)

participant_summary["unusual_percentage"] = (
    participant_summary["unusual_samples"]
    / participant_summary["total_samples"]
    * 100
)

participant_summary = participant_summary.sort_values(
    by="mean_score",
    ascending=False
)

print(
    participant_summary.to_string(index=False)
)

participant_summary.to_csv(
    OUTPUT_DIR / "participant_anomaly_summary.csv",
    index=False
)


# =========================================================
# 8. INTENSITY-WISE ANALYSIS
# =========================================================

print("\n" + "=" * 80)
print("INTENSITY-WISE ANALYSIS")
print("=" * 80)

intensity_summary = (
    df
    .groupby("intensity")
    .agg(
        total_samples=("is_unusual", "count"),
        unusual_samples=("is_unusual", "sum"),
        mean_score=("form_anomaly_score", "mean"),
        median_score=("form_anomaly_score", "median"),
        max_score=("form_anomaly_score", "max"),
    )
    .reset_index()
)

intensity_summary["unusual_percentage"] = (
    intensity_summary["unusual_samples"]
    / intensity_summary["total_samples"]
    * 100
)

print(
    intensity_summary.to_string(index=False)
)

intensity_summary.to_csv(
    OUTPUT_DIR / "intensity_anomaly_summary.csv",
    index=False
)


# =========================================================
# 9. PARTICIPANT + EXERCISE ANALYSIS
# =========================================================

print("\n" + "=" * 80)
print("PARTICIPANT + EXERCISE ANALYSIS")
print("=" * 80)

participant_exercise_summary = (
    df
    .groupby(
        [
            "participant",
            "exercise",
        ]
    )
    .agg(
        total_samples=("is_unusual", "count"),
        unusual_samples=("is_unusual", "sum"),
        mean_score=("form_anomaly_score", "mean"),
        max_score=("form_anomaly_score", "max"),
    )
    .reset_index()
)

participant_exercise_summary["unusual_percentage"] = (
    participant_exercise_summary["unusual_samples"]
    / participant_exercise_summary["total_samples"]
    * 100
)

participant_exercise_summary = (
    participant_exercise_summary
    .sort_values(
        by="mean_score",
        ascending=False
    )
)

print(
    participant_exercise_summary.to_string(index=False)
)

participant_exercise_summary.to_csv(
    OUTPUT_DIR
    / "participant_exercise_anomaly_summary.csv",
    index=False
)


# =========================================================
# 10. PARTICIPANT + EXERCISE + INTENSITY
# =========================================================

print("\n" + "=" * 80)
print("PARTICIPANT + EXERCISE + INTENSITY")
print("=" * 80)

detailed_summary = (
    df
    .groupby(
        [
            "participant",
            "exercise",
            "intensity",
        ]
    )
    .agg(
        total_samples=("is_unusual", "count"),
        unusual_samples=("is_unusual", "sum"),
        mean_score=("form_anomaly_score", "mean"),
        max_score=("form_anomaly_score", "max"),
    )
    .reset_index()
)

detailed_summary["unusual_percentage"] = (
    detailed_summary["unusual_samples"]
    / detailed_summary["total_samples"]
    * 100
)

detailed_summary = detailed_summary.sort_values(
    by="mean_score",
    ascending=False
)

print(
    detailed_summary.to_string(index=False)
)

detailed_summary.to_csv(
    OUTPUT_DIR
    / "detailed_anomaly_summary.csv",
    index=False
)


# =========================================================
# 11. TOP 30 MOST UNUSUAL WINDOWS
# =========================================================

print("\n" + "=" * 80)
print("TOP 30 MOST UNUSUAL WINDOWS")
print("=" * 80)

top_unusual = (
    df
    .sort_values(
        by="form_anomaly_score",
        ascending=False
    )
    .head(30)
    .copy()
)

top_columns = [
    "participant",
    "exercise",
    "intensity",
    "form_anomaly_prediction",
    "form_anomaly_score",
]

print(
    top_unusual[
        top_columns
    ].to_string(index=False)
)

top_unusual[
    top_columns
].to_csv(
    OUTPUT_DIR
    / "top_30_unusual_windows.csv",
    index=False
)


# =========================================================
# 12. TOP NORMAL / LOWEST-SCORE WINDOWS
# =========================================================

print("\n" + "=" * 80)
print("LOWEST ANOMALY SCORE WINDOWS")
print("=" * 80)

top_normal = (
    df
    .sort_values(
        by="form_anomaly_score",
        ascending=True
    )
    .head(20)
    .copy()
)

print(
    top_normal[
        top_columns
    ].to_string(index=False)
)

top_normal[
    top_columns
].to_csv(
    OUTPUT_DIR
    / "lowest_anomaly_score_windows.csv",
    index=False
)


# =========================================================
# 13. SCORE QUANTILES
# =========================================================

print("\n" + "=" * 80)
print("ANOMALY SCORE QUANTILES")
print("=" * 80)

quantiles = df[
    "form_anomaly_score"
].quantile(
    [
        0.00,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]
)

print(quantiles)


# =========================================================
# 14. HIGH-ANOMALY THRESHOLD ANALYSIS
# =========================================================

print("\n" + "=" * 80)
print("HIGH-ANOMALY THRESHOLD ANALYSIS")
print("=" * 80)

# These are analysis thresholds only.
# They are NOT form-error thresholds.

thresholds = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
]

threshold_rows = []

for threshold in thresholds:

    count = int(
        (
            df["form_anomaly_score"]
            >= threshold
        ).sum()
    )

    percentage = (
        count
        / total_samples
        * 100
    )

    threshold_rows.append(
        {
            "threshold": threshold,
            "samples_above_threshold": count,
            "percentage": percentage,
        }
    )

threshold_df = pd.DataFrame(
    threshold_rows
)

print(
    threshold_df.to_string(index=False)
)

threshold_df.to_csv(
    OUTPUT_DIR
    / "anomaly_threshold_analysis.csv",
    index=False
)


# =========================================================
# 15. PLOT 1 — OVERALL SCORE DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["form_anomaly_score"],
    bins=30
)

plt.xlabel("Anomaly Score")
plt.ylabel("Number of Windows")
plt.title(
    "Overall Form Anomaly Score Distribution"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "overall_anomaly_distribution.png",
    dpi=150
)

plt.close()


# =========================================================
# 16. PLOT 2 — EXERCISE MEAN SCORE
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    exercise_summary["exercise"],
    exercise_summary["mean_score"]
)

plt.xlabel("Exercise")
plt.ylabel("Mean Anomaly Score")
plt.title(
    "Mean Anomaly Score by Exercise"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "exercise_mean_anomaly_score.png",
    dpi=150
)

plt.close()


# =========================================================
# 17. PLOT 3 — PARTICIPANT UNUSUAL RATE
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    participant_summary["participant"],
    participant_summary["unusual_percentage"]
)

plt.xlabel("Participant")
plt.ylabel("Unusual Windows (%)")
plt.title(
    "Unusual Window Rate by Participant"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "participant_unusual_rate.png",
    dpi=150
)

plt.close()


# =========================================================
# 18. PLOT 4 — INTENSITY MEAN SCORE
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    intensity_summary["intensity"],
    intensity_summary["mean_score"]
)

plt.xlabel("Intensity / State")
plt.ylabel("Mean Anomaly Score")
plt.title(
    "Mean Anomaly Score by Intensity / State"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "intensity_mean_anomaly_score.png",
    dpi=150
)

plt.close()


# =========================================================
# 19. PLOT 5 — EXERCISE SCORE BOX PLOT
# =========================================================

exercise_order = sorted(
    df["exercise"].unique()
)

box_data = [
    df.loc[
        df["exercise"] == exercise,
        "form_anomaly_score"
    ].values
    for exercise in exercise_order
]

plt.figure(figsize=(10, 6))

plt.boxplot(
    box_data,
    tick_labels=exercise_order
)

plt.xlabel("Exercise")
plt.ylabel("Anomaly Score")
plt.title(
    "Anomaly Score Distribution by Exercise"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    PLOT_DIR
    / "exercise_anomaly_boxplot.png",
    dpi=150
)

plt.close()


# =========================================================
# 20. FIND REPEATED HIGH-ANOMALY GROUPS
# =========================================================

print("\n" + "=" * 80)
print("REPEATED HIGH-ANOMALY GROUPS")
print("=" * 80)

high_anomaly_df = df[
    df["form_anomaly_score"] >= 0.08
].copy()

if high_anomaly_df.empty:

    print(
        "No windows crossed the analysis threshold 0.08."
    )

else:

    repeated_groups = (
        high_anomaly_df
        .groupby(
            [
                "participant",
                "exercise",
                "intensity",
            ]
        )
        .agg(
            high_anomaly_windows=(
                "form_anomaly_score",
                "count"
            ),
            mean_score=(
                "form_anomaly_score",
                "mean"
            ),
            max_score=(
                "form_anomaly_score",
                "max"
            ),
        )
        .reset_index()
        .sort_values(
            by=[
                "high_anomaly_windows",
                "max_score",
            ],
            ascending=False
        )
    )

    print(
        repeated_groups.to_string(
            index=False
        )
    )

    repeated_groups.to_csv(
        OUTPUT_DIR
        / "repeated_high_anomaly_groups.csv",
        index=False
    )


# =========================================================
# 21. FINAL REPORT
# =========================================================

final_report = {
    "total_windows": int(total_samples),
    "normal_windows": int(normal_samples),
    "unusual_windows": int(unusual_samples),
    "unusual_percentage": round(
        overall_unusual_percentage,
        4
    ),
    "mean_anomaly_score": round(
        float(df["form_anomaly_score"].mean()),
        6
    ),
    "median_anomaly_score": round(
        float(df["form_anomaly_score"].median()),
        6
    ),
    "minimum_anomaly_score": round(
        float(df["form_anomaly_score"].min()),
        6
    ),
    "maximum_anomaly_score": round(
        float(df["form_anomaly_score"].max()),
        6
    ),
}

final_report_df = pd.DataFrame(
    [
        final_report
    ]
)

final_report_df.to_csv(
    OUTPUT_DIR
    / "form_anomaly_final_report.csv",
    index=False
)


# =========================================================
# 22. FINAL MESSAGE
# =========================================================

print("\n" + "=" * 80)
print("ANALYSIS FILES SAVED")
print("=" * 80)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    f"\nPlots directory:\n{PLOT_DIR}"
)

print(
    """
Created files:

1. exercise_anomaly_summary.csv
2. participant_anomaly_summary.csv
3. intensity_anomaly_summary.csv
4. participant_exercise_anomaly_summary.csv
5. detailed_anomaly_summary.csv
6. top_30_unusual_windows.csv
7. lowest_anomaly_score_windows.csv
8. anomaly_threshold_analysis.csv
9. repeated_high_anomaly_groups.csv
10. form_anomaly_final_report.csv
"""
)

print(
    """
IMPORTANT:
The anomaly score indicates deviation from the learned
movement pattern.

It does NOT prove that the exercise form is incorrect,
because the dataset has no ground-truth form-error labels.
"""
)

print("\nFORM ANOMALY ANALYSIS COMPLETED ✅")
print("=" * 80)