from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_feature_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. SETTINGS
# =========================================================

RANDOM_STATE = 42

# Approximate fraction of unusual samples.
# This is NOT a form-error ground-truth percentage.
CONTAMINATION = 0.10


# =========================================================
# 3. LOAD DATA
# =========================================================

print("=" * 80)
print("AI FITNESS TRACKER")
print("FORM ANOMALY DETECTION")
print("=" * 80)

print("\nLoading feature dataset...")

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found:\n{FEATURE_FILE}"
    )

df = pd.read_csv(FEATURE_FILE)

print(f"Dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# 4. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "participant",
    "exercise",
    "intensity",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =========================================================
# 5. REMOVE REST
# =========================================================

working_df = df[
    df["exercise"] != "rest"
].copy()

print("\nAfter removing rest:")
print(f"Shape: {working_df.shape}")


# =========================================================
# 6. SELECT NUMERIC SENSOR FEATURES
# =========================================================

exclude_columns = {
    "participant",
    "exercise",
    "intensity",
    "source_file",
    "window_id",
}

feature_columns = [
    col
    for col in working_df.columns
    if col not in exclude_columns
    and pd.api.types.is_numeric_dtype(working_df[col])
]

if not feature_columns:
    raise ValueError(
        "No numeric sensor features found."
    )

print("\nNumber of sensor features:")
print(len(feature_columns))

print("\nFeature columns:")
for col in feature_columns:
    print(f"  {col}")


# =========================================================
# 7. HANDLE MISSING VALUES
# =========================================================

missing_before = working_df[
    feature_columns
].isna().sum().sum()

print("\nMissing feature values before cleaning:")
print(missing_before)

if missing_before > 0:

    working_df[feature_columns] = (
        working_df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    working_df[feature_columns] = (
        working_df[feature_columns]
        .fillna(
            working_df[feature_columns].median()
        )
    )

missing_after = working_df[
    feature_columns
].isna().sum().sum()

print("Missing feature values after cleaning:")
print(missing_after)


# =========================================================
# 8. TRAIN ONE ANOMALY MODEL PER EXERCISE
# =========================================================

exercise_list = sorted(
    working_df["exercise"].dropna().unique()
)

print("\nExercises:")
print(exercise_list)


all_results = []

model_metadata = {}


for exercise in exercise_list:

    print("\n" + "=" * 80)
    print(f"EXERCISE: {exercise.upper()}")
    print("=" * 80)

    exercise_df = working_df[
        working_df["exercise"] == exercise
    ].copy()

    print(
        f"Samples: {len(exercise_df)}"
    )

    # -----------------------------------------------------
    # Feature matrix
    # -----------------------------------------------------

    X = exercise_df[
        feature_columns
    ].values

    # -----------------------------------------------------
    # Scale
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # Isolation Forest
    # -----------------------------------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # -----------------------------------------------------
    # Predictions
    #
    # 1  = normal
    # -1 = anomaly
    # -----------------------------------------------------

    predictions = model.predict(X_scaled)

    # -----------------------------------------------------
    # Raw anomaly score
    #
    # Larger positive value = more normal
    # Smaller value = more unusual
    # -----------------------------------------------------

    normality_score = model.decision_function(
        X_scaled
    )

    # Convert to intuitive anomaly score
    anomaly_score = -normality_score

    exercise_df["form_anomaly_prediction"] = np.where(
        predictions == -1,
        "unusual",
        "normal",
    )

    exercise_df["form_anomaly_score"] = anomaly_score

    # -----------------------------------------------------
    # Store results
    # -----------------------------------------------------

    all_results.append(
        exercise_df
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_file = (
        MODEL_DIR
        / f"form_{exercise}_isolation_forest.pkl"
    )

    scaler_file = (
        MODEL_DIR
        / f"form_{exercise}_scaler.pkl"
    )

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    anomaly_count = int(
        np.sum(predictions == -1)
    )

    normal_count = int(
        np.sum(predictions == 1)
    )

    anomaly_percentage = (
        anomaly_count
        / len(predictions)
        * 100
    )

    print(
        f"Normal samples   : {normal_count}"
    )

    print(
        f"Unusual samples  : {anomaly_count}"
    )

    print(
        f"Unusual %        : "
        f"{anomaly_percentage:.2f}%"
    )

    print(
        f"Min anomaly score: "
        f"{anomaly_score.min():.4f}"
    )

    print(
        f"Max anomaly score: "
        f"{anomaly_score.max():.4f}"
    )

    print(
        f"Mean score       : "
        f"{anomaly_score.mean():.4f}"
    )

    model_metadata[exercise] = {
        "samples": int(len(exercise_df)),
        "features": int(len(feature_columns)),
        "contamination": CONTAMINATION,
        "normal_samples": normal_count,
        "unusual_samples": anomaly_count,
        "unusual_percentage": round(
            anomaly_percentage,
            4
        ),
        "model_file": str(model_file),
        "scaler_file": str(scaler_file),
    }


# =========================================================
# 9. COMBINE RESULTS
# =========================================================

results_df = pd.concat(
    all_results,
    ignore_index=True
)


# =========================================================
# 10. SORT BY ANOMALY SCORE
# =========================================================

results_df = results_df.sort_values(
    by="form_anomaly_score",
    ascending=False
)


# =========================================================
# 11. SAVE ALL RESULTS
# =========================================================

result_file = (
    OUTPUT_DIR
    / "form_anomaly_scores.csv"
)

results_df.to_csv(
    result_file,
    index=False
)


# =========================================================
# 12. SAVE METADATA
# =========================================================

metadata_file = (
    OUTPUT_DIR
    / "form_anomaly_metadata.json"
)

metadata = {
    "method": "Isolation Forest",
    "purpose": (
        "Unsupervised movement anomaly detection"
    ),
    "warning": (
        "An unusual prediction does not mean "
        "incorrect form because the dataset does "
        "not contain ground-truth form labels."
    ),
    "dataset_shape": list(df.shape),
    "working_shape": list(working_df.shape),
    "feature_count": len(feature_columns),
    "feature_columns": feature_columns,
    "random_state": RANDOM_STATE,
    "contamination": CONTAMINATION,
    "models": model_metadata,
}

with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# =========================================================
# 13. TOP UNUSUAL WINDOWS
# =========================================================

print("\n" + "=" * 80)
print("TOP 20 MOST UNUSUAL WINDOWS")
print("=" * 80)

display_columns = [
    "participant",
    "exercise",
    "intensity",
    "form_anomaly_prediction",
    "form_anomaly_score",
]

print(
    results_df[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# =========================================================
# 14. EXERCISE SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("EXERCISE ANOMALY SUMMARY")
print("=" * 80)

summary = (
    results_df
    .groupby("exercise")
    .agg(
        total_samples=(
            "form_anomaly_prediction",
            "count"
        ),
        unusual_samples=(
            "form_anomaly_prediction",
            lambda x: (
                x == "unusual"
            ).sum()
        ),
        mean_anomaly_score=(
            "form_anomaly_score",
            "mean"
        ),
    )
    .reset_index()
)

summary["unusual_percentage"] = (
    summary["unusual_samples"]
    / summary["total_samples"]
    * 100
)

print(
    summary.to_string(index=False)
)


# =========================================================
# 15. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"\nAnomaly results:\n{result_file}"
)

print(
    f"\nMetadata:\n{metadata_file}"
)

print(
    f"\nModels:\n{MODEL_DIR}"
)


# =========================================================
# 16. FINAL NOTE
# =========================================================

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    """
This module performs UNSUPERVISED movement anomaly detection.

'normal'   = movement is close to the learned exercise pattern
'unusual'  = movement is relatively different from the pattern

'unusual' MUST NOT be interpreted as 'incorrect form'
until actual form-error ground-truth labels are available.
"""
)

print("\nFORM ANOMALY DETECTOR COMPLETED ✅")
print("=" * 80)