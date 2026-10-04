from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_DIR = (
    PROJECT_ROOT /
    "data" /
    "features"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "models" /
    "saved_models"
)

DATA_FILE = (
    FEATURES_DIR /
    "final_feature_dataset.csv"
)

MODEL_FILE = (
    MODEL_DIR /
    "exercise_classifier.pkl"
)

FEATURE_LIST_FILE = (
    MODEL_DIR /
    "exercise_features.pkl"
)

CLASSES_FILE = (
    MODEL_DIR /
    "exercise_classes.pkl"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

def load_model():
    """
    Load trained exercise classification model.
    """

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model loaded:\n{MODEL_FILE}"
    )

    return model


# ============================================================
# 3. LOAD FEATURE LIST
# ============================================================

def load_feature_list():
    """
    Load the exact feature names/order used
    during model training.
    """

    if not FEATURE_LIST_FILE.exists():

        raise FileNotFoundError(
            f"Feature list not found:\n"
            f"{FEATURE_LIST_FILE}"
        )

    feature_columns = joblib.load(
        FEATURE_LIST_FILE
    )

    print(
        f"Feature list loaded."
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    return feature_columns


# ============================================================
# 4. LOAD CLASSES
# ============================================================

def load_classes():
    """
    Load model class labels.
    """

    if not CLASSES_FILE.exists():

        raise FileNotFoundError(
            f"Classes file not found:\n"
            f"{CLASSES_FILE}"
        )

    classes = joblib.load(
        CLASSES_FILE
    )

    print(
        f"Classes: {classes}"
    )

    return classes


# ============================================================
# 5. LOAD FEATURE DATASET
# ============================================================

def load_dataset():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\nDataset loaded."
    )

    print(
        f"Shape: {df.shape}"
    )

    return df


# ============================================================
# 6. PREPARE INPUT
# ============================================================

def prepare_input(
    row,
    feature_columns
):
    """
    Extract features from one dataframe row
    in exactly the same order as training.
    """

    missing_columns = [
        column
        for column in feature_columns
        if column not in row.index
    ]

    if missing_columns:

        raise ValueError(
            "Required feature columns missing:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Select exact feature order
    # --------------------------------------------------------

    X = row[
        feature_columns
    ].copy()

    # Convert to numeric
    X = pd.to_numeric(
        X,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    if X.isna().any():

        missing = X[
            X.isna()
        ].index.tolist()

        raise ValueError(
            "Missing/non-numeric values found in:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Check infinite values
    # --------------------------------------------------------

    if not np.isfinite(
        X.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Infinite value found in input."
        )

    # --------------------------------------------------------
    # Model expects 2D input
    # --------------------------------------------------------

    X = X.to_frame().T

    return X


# ============================================================
# 7. PREDICT ONE SAMPLE
# ============================================================

def predict_exercise(
    model,
    row,
    feature_columns
):
    """
    Predict exercise and confidence for one sample.
    """

    X = prepare_input(
        row,
        feature_columns
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X
    )[0]

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X
    )[0]

    class_labels = model.classes_

    # Find probability of predicted class
    predicted_index = list(
        class_labels
    ).index(
        prediction
    )

    confidence = (
        probabilities[
            predicted_index
        ] * 100
    )

    # --------------------------------------------------------
    # Top 3 predictions
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    top_predictions = []

    for index in sorted_indices[:3]:

        top_predictions.append(
            (
                class_labels[index],
                probabilities[index] * 100
            )
        )

    return (
        prediction,
        confidence,
        top_predictions
    )


# ============================================================
# 8. DISPLAY PREDICTION
# ============================================================

def display_prediction(
    row,
    prediction,
    confidence,
    top_predictions
):

    print("\n" + "=" * 60)
    print("EXERCISE PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Actual label
    # --------------------------------------------------------

    if "exercise" in row.index:

        print(
            f"\nActual exercise : "
            f"{row['exercise']}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if "participant" in row.index:

        print(
            f"Participant     : "
            f"{row['participant']}"
        )

    if "intensity" in row.index:

        print(
            f"Intensity       : "
            f"{row['intensity']}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print(
        f"\nPredicted       : "
        f"{prediction}"
    )

    print(
        f"Confidence      : "
        f"{confidence:.2f}%"
    )

    # --------------------------------------------------------
    # Top 3
    # --------------------------------------------------------

    print(
        "\nTop 3 predictions:"
    )

    for exercise, probability in top_predictions:

        print(
            f"  {exercise:<10} "
            f"{probability:.2f}%"
        )

    # --------------------------------------------------------
    # Correct/incorrect
    # --------------------------------------------------------

    if "exercise" in row.index:

        actual = str(
            row["exercise"]
        )

        predicted = str(
            prediction
        )

        if actual == predicted:

            print(
                "\nPrediction: CORRECT ✅"
            )

        else:

            print(
                "\nPrediction: INCORRECT ❌"
            )


# ============================================================
# 9. TEST ONE SAMPLE
# ============================================================

def test_single_sample(
    model,
    feature_columns,
    df,
    row_index=0
):

    if row_index < 0 or row_index >= len(df):

        raise IndexError(
            f"row_index must be between "
            f"0 and {len(df) - 1}"
        )

    row = df.iloc[
        row_index
    ]

    (
        prediction,
        confidence,
        top_predictions
    ) = predict_exercise(
        model=model,
        row=row,
        feature_columns=feature_columns
    )

    display_prediction(
        row=row,
        prediction=prediction,
        confidence=confidence,
        top_predictions=top_predictions
    )


# ============================================================
# 10. TEST RANDOM SAMPLES
# ============================================================

def test_random_samples(
    model,
    feature_columns,
    df,
    number_of_samples=10
):

    number_of_samples = min(
        number_of_samples,
        len(df)
    )

    # --------------------------------------------------------
    # Fixed random seed for reproducibility
    # --------------------------------------------------------

    samples = df.sample(
        n=number_of_samples,
        random_state=42
    )

    correct = 0

    print("\n" + "=" * 60)
    print("RANDOM SAMPLE PREDICTIONS")
    print("=" * 60)

    for index, row in samples.iterrows():

        (
            prediction,
            confidence,
            top_predictions
        ) = predict_exercise(
            model=model,
            row=row,
            feature_columns=feature_columns
        )

        actual = row["exercise"]

        is_correct = (
            actual == prediction
        )

        if is_correct:

            correct += 1

        print(
            f"\nRow {index}"
        )

        print(
            f"Actual     : {actual}"
        )

        print(
            f"Predicted  : {prediction}"
        )

        print(
            f"Confidence : {confidence:.2f}%"
        )

        print(
            f"Result     : "
            f"{'CORRECT ✅' if is_correct else 'WRONG ❌'}"
        )

    accuracy = (
        correct /
        number_of_samples
    ) * 100

    print("\n" + "=" * 60)

    print(
        f"Random sample accuracy: "
        f"{accuracy:.2f}%"
    )

    print("=" * 60)


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER")
    print("EXERCISE DETECTOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    model = load_model()

    feature_columns = (
        load_feature_list()
    )

    classes = load_classes()

    df = load_dataset()

    # --------------------------------------------------------
    # Verify dataset features
    # --------------------------------------------------------

    dataset_features = [
        column
        for column in df.columns
        if column not in [
            "participant",
            "exercise",
            "intensity"
        ]
    ]

    missing_features = [
        column
        for column in feature_columns
        if column not in dataset_features
    ]

    if missing_features:

        raise ValueError(
            "Some trained features are missing "
            "from final dataset:\n"
            + "\n".join(missing_features)
        )

    # --------------------------------------------------------
    # Model class check
    # --------------------------------------------------------

    print(
        f"\nModel classes:"
    )

    print(
        list(classes)
    )

    # --------------------------------------------------------
    # Test one sample
    # --------------------------------------------------------

    test_single_sample(
        model=model,
        feature_columns=feature_columns,
        df=df,
        row_index=0
    )

    # --------------------------------------------------------
    # Test multiple random samples
    # --------------------------------------------------------

    test_random_samples(
        model=model,
        feature_columns=feature_columns,
        df=df,
        number_of_samples=10
    )

    print("\n" + "=" * 60)
    print("EXERCISE DETECTOR TEST COMPLETED ✅")
    print("=" * 60)


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":

    main()