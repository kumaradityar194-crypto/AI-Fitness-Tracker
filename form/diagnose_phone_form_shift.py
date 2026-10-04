from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_feature_dataset.csv"
)

PHONE_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_final_feature_dataset.csv"
)

PHONE_FORM_RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_form_anomaly_scores.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "form_anomaly_metadata.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "form_squat_isolation_forest.pkl"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "form_squat_scaler.pkl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_form_shift_diagnostic.csv"
)


def load_feature_columns():
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
            "Form feature columns were not found."
        )

    return feature_columns


def load_model():
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


def calculate_scores(
    df,
    feature_columns,
    model,
    scaler,
):
    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    working_df = df.copy()

    for column in feature_columns:
        working_df[column] = pd.to_numeric(
            working_df[column],
            errors="coerce",
        )

    working_df[feature_columns] = (
        working_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    working_df = working_df.dropna(
        subset=feature_columns
    )

    X = working_df[
        feature_columns
    ].values

    X_scaled = scaler.transform(
        X
    )

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

    return (
        working_df,
        predictions,
        anomaly_scores,
    )


def percentile_rank(
    reference_scores,
    score,
):
    return float(
        np.mean(
            reference_scores <= score
        )
        * 100.0
    )


def main():
    print()
    print("=" * 78)
    print("AI FITNESS TRACKER - PHONE FORM SHIFT DIAGNOSTIC")
    print("=" * 78)

    print()
    print("Loading metadata...")

    feature_columns = load_feature_columns()

    print(
        f"Feature count: {len(feature_columns)}"
    )

    print()
    print("Loading saved squat form model...")

    model, scaler = load_model()

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Scaler: {SCALER_FILE}"
    )

    print()
    print("Loading training dataset...")

    train_df = pd.read_csv(
        TRAIN_FEATURE_FILE
    )

    train_squat_df = train_df[
        train_df["exercise"]
        .astype(str)
        .str.lower()
        .str.strip()
        == "squat"
    ].copy()

    print(
        f"Training squat rows: "
        f"{len(train_squat_df)}"
    )

    print()
    print("Scoring training squat data...")

    (
        train_clean,
        train_predictions,
        train_scores,
    ) = calculate_scores(
        train_squat_df,
        feature_columns,
        model,
        scaler,
    )

    print(
        f"Training rows scored: "
        f"{len(train_clean)}"
    )

    print()
    print("Loading phone form results...")

    phone_result_df = pd.read_csv(
        PHONE_FORM_RESULT_FILE
    )

    phone_features_df = pd.read_csv(
        PHONE_FEATURE_FILE
    )

    print(
        f"Phone windows: "
        f"{len(phone_features_df)}"
    )

    print()
    print("Scoring phone data...")

    (
        phone_clean,
        phone_predictions,
        phone_scores,
    ) = calculate_scores(
        phone_features_df,
        feature_columns,
        model,
        scaler,
    )

    print(
        f"Phone rows scored: "
        f"{len(phone_clean)}"
    )

    training_unusual = int(
        np.sum(
            train_predictions == -1
        )
    )

    phone_unusual = int(
        np.sum(
            phone_predictions == -1
        )
    )

    training_unusual_percentage = (
        training_unusual
        / len(train_predictions)
        * 100
    )

    phone_unusual_percentage = (
        phone_unusual
        / len(phone_predictions)
        * 100
    )

    print()
    print("=" * 78)
    print("TRAINING SCORE DISTRIBUTION")
    print("=" * 78)

    print(
        f"Minimum score : "
        f"{train_scores.min():.6f}"
    )

    print(
        f"10th percentile: "
        f"{np.percentile(train_scores, 10):.6f}"
    )

    print(
        f"25th percentile: "
        f"{np.percentile(train_scores, 25):.6f}"
    )

    print(
        f"Median score  : "
        f"{np.median(train_scores):.6f}"
    )

    print(
        f"75th percentile: "
        f"{np.percentile(train_scores, 75):.6f}"
    )

    print(
        f"90th percentile: "
        f"{np.percentile(train_scores, 90):.6f}"
    )

    print(
        f"Maximum score : "
        f"{train_scores.max():.6f}"
    )

    print(
        f"Unusual rows  : "
        f"{training_unusual}"
    )

    print(
        f"Unusual rate  : "
        f"{training_unusual_percentage:.2f}%"
    )

    print()
    print("=" * 78)
    print("PHONE SCORE DISTRIBUTION")
    print("=" * 78)

    print(
        f"Minimum score : "
        f"{phone_scores.min():.6f}"
    )

    print(
        f"Median score  : "
        f"{np.median(phone_scores):.6f}"
    )

    print(
        f"Maximum score : "
        f"{phone_scores.max():.6f}"
    )

    print(
        f"Unusual rows  : "
        f"{phone_unusual}"
    )

    print(
        f"Unusual rate  : "
        f"{phone_unusual_percentage:.2f}%"
    )

    diagnostic_rows = []

    for index in range(
        len(phone_clean)
    ):
        window_id = int(
            phone_clean.iloc[index][
                "window_id"
            ]
        )

        score = float(
            phone_scores[index]
        )

        prediction = (
            "unusual"
            if phone_predictions[index] == -1
            else "normal"
        )

        percentile = percentile_rank(
            train_scores,
            score,
        )

        diagnostic_rows.append(
            {
                "window_id": window_id,
                "form_prediction": prediction,
                "phone_anomaly_score": score,
                "training_score_percentile": percentile,
            }
        )

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    if not phone_result_df.empty:
        confidence_columns = [
            "window_id",
            "confidence_percent",
        ]

        available_columns = [
            column
            for column in confidence_columns
            if column in phone_result_df.columns
        ]

        if "window_id" in available_columns:
            diagnostic_df = diagnostic_df.merge(
                phone_result_df[
                    available_columns
                ],
                on="window_id",
                how="left",
            )

    diagnostic_df = diagnostic_df.sort_values(
        "window_id"
    ).reset_index(
        drop=True
    )

    diagnostic_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 78)
    print("PHONE WINDOW DIAGNOSTIC")
    print("=" * 78)

    print(
        diagnostic_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print("=" * 78)
    print("PHONE FORM SHIFT DIAGNOSTIC COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()