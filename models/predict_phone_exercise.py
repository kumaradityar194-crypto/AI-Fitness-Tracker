from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

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

FEATURE_LIST_FILE = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "exercise_features_phone_adapted.pkl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_exercise_predictions.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI FITNESS TRACKER - PHONE EXERCISE PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_FILE}"
        )

    if not FEATURE_LIST_FILE.exists():
        raise FileNotFoundError(
            f"Feature list not found:\n{FEATURE_LIST_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        "\nPhone feature dataset:",
        df.shape
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Load exact feature order
    # --------------------------------------------------------

    feature_names = joblib.load(
        FEATURE_LIST_FILE
    )

    print(
        "Model features:",
        len(feature_names)
    )

    # --------------------------------------------------------
    # Verify features
    # --------------------------------------------------------

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:

        print("\nMissing features:")

        for feature in missing:
            print(
                " ",
                feature
            )

        raise ValueError(
            "Phone dataset does not contain "
            "all model features."
        )

    # --------------------------------------------------------
    # EXACT SAME FEATURE ORDER
    # --------------------------------------------------------

    X = df[
        feature_names
    ].copy()

    print(
        "Prediction input shape:",
        X.shape
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(X)

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = (
            model.predict_proba(X)
        )

        confidence = (
            probabilities.max(axis=1)
            * 100
        )

    else:

        confidence = [
            None
            for _ in predictions
        ]

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result = pd.DataFrame({
        "window_id": df["window_id"],
        "predicted_exercise": predictions,
        "confidence_percent": confidence
    })

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Show results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)

    print("\nFirst 20 predictions:")

    print(
        result.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print(
        "\nPrediction distribution:"
    )

    print(
        result["predicted_exercise"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Average confidence
    # --------------------------------------------------------

    if result["confidence_percent"].notna().any():

        print(
            "\nAverage confidence:",
            round(
                result[
                    "confidence_percent"
                ].mean(),
                2
            ),
            "%"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 70)
    print("PHONE EXERCISE PREDICTION COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()