from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_final_feature_dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "exercise_classifier_phone_adapted.pkl"
)

FEATURE_NAMES_FILE = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "exercise_features_phone_adapted.pkl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_prediction_diagnostics.csv"
)


def main():
    print("=" * 78)
    print("AI FITNESS TRACKER - PHONE PREDICTION DIAGNOSTICS")
    print("=" * 78)

    print("\nLoading model and feature data...")

    model = joblib.load(MODEL_FILE)
    feature_names = list(joblib.load(FEATURE_NAMES_FILE))
    phone_df = pd.read_csv(FEATURE_FILE)

    missing = [c for c in feature_names if c not in phone_df.columns]

    if missing:
        raise ValueError(f"Missing phone features: {missing}")

    x = phone_df[feature_names].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if x.isna().any().any():
        raise ValueError("Phone feature dataset contains invalid values.")

    print(f"Phone dataset shape: {phone_df.shape}")
    print(f"Prediction input shape: {x.shape}")

    predictions = model.predict(x)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        classes = list(model.classes_)
    else:
        raise ValueError(
            "The loaded classifier does not provide predict_proba()."
        )

    result = pd.DataFrame(
        {
            "window_id": phone_df["window_id"],
            "predicted_exercise": predictions,
        }
    )

    for idx, cls in enumerate(classes):
        result[f"prob_{cls}"] = probabilities[:, idx]

    result["confidence_percent"] = (
        np.max(probabilities, axis=1) * 100.0
    )

    print("\n" + "=" * 78)
    print("WINDOW-LEVEL PROBABILITIES")
    print("=" * 78)

    display_columns = [
        "window_id",
        "predicted_exercise",
    ]

    display_columns += [
        f"prob_{cls}"
        for cls in classes
    ]

    display_columns.append("confidence_percent")

    display_df = result[display_columns].copy()

    for cls in classes:
        display_df[f"prob_{cls}"] = (
            display_df[f"prob_{cls}"] * 100.0
        ).round(2)

    display_df["confidence_percent"] = display_df[
        "confidence_percent"
    ].round(2)

    print(display_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("AVERAGE CLASS PROBABILITY")
    print("=" * 78)

    probability_columns = [
        f"prob_{cls}"
        for cls in classes
    ]

    avg_probabilities = (
        result[probability_columns]
        .mean()
        .sort_values(ascending=False)
    )

    for column, value in avg_probabilities.items():
        exercise = column.replace("prob_", "")
        print(f"{exercise:10s}: {value * 100:.2f}%")

    print("\n" + "=" * 78)
    print("PREDICTION DISTRIBUTION")
    print("=" * 78)

    distribution = (
        result["predicted_exercise"]
        .value_counts()
        .sort_index()
    )

    for exercise, count in distribution.items():
        print(f"{exercise:10s}: {count}")

    print("\n" + "=" * 78)
    print("TEMPORAL CONSISTENCY")
    print("=" * 78)

    predicted_values = result["predicted_exercise"].tolist()

    switches = sum(
        1
        for i in range(1, len(predicted_values))
        if predicted_values[i] != predicted_values[i - 1]
    )

    consistency = (
        1.0
        if len(predicted_values) <= 1
        else 1.0 - switches / (len(predicted_values) - 1)
    )

    print(f"Prediction switches: {switches}")
    print(f"Temporal consistency: {consistency * 100:.2f}%")

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 78)
    print("PHONE PREDICTION DIAGNOSTICS COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()