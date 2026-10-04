from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_final_feature_dataset.csv"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_exercise_predictions.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "form_anomaly_metadata.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
)

MODEL_FILE = (
    MODEL_DIR
    / "form_squat_isolation_forest.pkl"
)

SCALER_FILE = (
    MODEL_DIR
    / "form_squat_scaler.pkl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_form_anomaly_scores.csv"
)


def load_feature_columns():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Form metadata file not found: {METADATA_FILE}"
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    feature_columns = metadata.get(
        "feature_columns"
    )

    if not feature_columns:
        raise ValueError(
            "Form feature columns were not found in metadata."
        )

    return feature_columns


def load_phone_features(feature_columns):
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Phone feature file not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    if "window_id" not in df.columns:
        raise ValueError(
            "Phone feature dataset does not contain window_id."
        )

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing form feature columns: {missing_columns}"
        )

    for column in feature_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    missing_values = (
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"Phone form features contain "
            f"{missing_values} missing values."
        )

    return df


def load_exercise_predictions():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Exercise prediction file not found: "
            f"{PREDICTION_FILE}"
        )

    prediction_df = pd.read_csv(
        PREDICTION_FILE
    )

    required_columns = [
        "window_id",
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
            f"Missing exercise prediction columns: "
            f"{missing_columns}"
        )

    prediction_df["predicted_exercise"] = (
        prediction_df["predicted_exercise"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    prediction_df["confidence_percent"] = pd.to_numeric(
        prediction_df["confidence_percent"],
        errors="coerce",
    )

    return prediction_df


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Squat form model not found: {MODEL_FILE}"
        )

    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Squat form scaler not found: {SCALER_FILE}"
        )

    with open(
        MODEL_FILE,
        "rb",
    ) as file:
        model = pickle.load(file)

    with open(
        SCALER_FILE,
        "rb",
    ) as file:
        scaler = pickle.load(file)

    return model, scaler


def main():
    print()
    print("=" * 75)
    print("AI FITNESS TRACKER - PHONE FORM ANOMALY DETECTOR")
    print("=" * 75)

    print()
    print("Loading form feature metadata...")

    feature_columns = load_feature_columns()

    print(
        f"Form feature count: {len(feature_columns)}"
    )

    print()
    print("Loading phone feature dataset...")

    feature_df = load_phone_features(
        feature_columns
    )

    print(
        f"Phone windows: {len(feature_df)}"
    )

    print()
    print("Loading exercise predictions...")

    prediction_df = load_exercise_predictions()

    print(
        f"Prediction rows: {len(prediction_df)}"
    )

    print()
    print("Checking predicted exercise...")

    exercise_distribution = (
        prediction_df["predicted_exercise"]
        .value_counts()
    )

    print(
        exercise_distribution.to_string()
    )

    if not (
        prediction_df["predicted_exercise"]
        .eq("squat")
        .all()
    ):
        raise ValueError(
            "Phone prediction dataset contains "
            "non-squat predictions. "
            "This detector is configured for squat."
        )

    print()
    print("Loading squat form model...")

    model, scaler = load_model()

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Scaler: {SCALER_FILE}"
    )

    X = feature_df[
        feature_columns
    ].values

    print()
    print("Scaling phone features...")

    X_scaled = scaler.transform(
        X
    )

    print(
        f"Scaled matrix shape: {X_scaled.shape}"
    )

    print()
    print("Running Isolation Forest...")

    predictions = model.predict(
        X_scaled
    )

    normality_scores = (
        model.decision_function(
            X_scaled
        )
    )

    anomaly_scores = (
        -normality_scores
    )

    result_df = feature_df[
        ["window_id"]
    ].copy()

    result_df["exercise"] = (
        "squat"
    )

    result_df["form_anomaly_prediction"] = np.where(
        predictions == -1,
        "unusual",
        "normal",
    )

    result_df["form_anomaly_score"] = (
        anomaly_scores
    )

    result_df = result_df.merge(
        prediction_df[
            [
                "window_id",
                "confidence_percent",
            ]
        ],
        on="window_id",
        how="left",
    )

    result_df = result_df.sort_values(
        "window_id"
    ).reset_index(
        drop=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    normal_count = int(
        np.sum(
            predictions == 1
        )
    )

    unusual_count = int(
        np.sum(
            predictions == -1
        )
    )

    unusual_percentage = (
        unusual_count
        / len(predictions)
        * 100
    )

    print()
    print("=" * 75)
    print("PHONE FORM ANOMALY RESULT")
    print("=" * 75)

    print(
        f"Total windows       : {len(result_df)}"
    )

    print(
        f"Normal windows      : {normal_count}"
    )

    print(
        f"Unusual windows     : {unusual_count}"
    )

    print(
        f"Unusual percentage  : "
        f"{unusual_percentage:.2f}%"
    )

    print(
        f"Mean anomaly score  : "
        f"{anomaly_scores.mean():.6f}"
    )

    print(
        f"Minimum score       : "
        f"{anomaly_scores.min():.6f}"
    )

    print(
        f"Maximum score       : "
        f"{anomaly_scores.max():.6f}"
    )

    print()
    print("Window-level results:")
    print()

    print(
        result_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print("=" * 75)
    print("PHONE FORM ANOMALY DETECTION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()