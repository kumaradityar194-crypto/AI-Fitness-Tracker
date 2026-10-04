from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_FILE = (
    FEATURES_DIR /
    "final_feature_dataset.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 60)
    print("LOADING FINAL FEATURE DATASET")
    print("=" * 60)

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\nDataset shape: {df.shape}"
    )

    print(
        "\nColumns:"
    )

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# 3. PREPARE X, y, GROUPS
# ============================================================

def prepare_data(df):

    print("\n" + "=" * 60)
    print("PREPARING ML DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if "exercise" not in df.columns:

        raise ValueError(
            "exercise column not found."
        )

    y = df["exercise"].copy()

    # --------------------------------------------------------
    # Participant is used ONLY as grouping variable.
    # It must not enter X.
    # --------------------------------------------------------

    if "participant" not in df.columns:

        raise ValueError(
            "participant column not found."
        )

    groups = df["participant"].copy()

    # --------------------------------------------------------
    # Only sensor-derived numeric features
    # --------------------------------------------------------

    metadata_columns = {
        "participant",
        "exercise",
        "intensity"
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    # --------------------------------------------------------
    # Remove any accidental ID/string columns
    # --------------------------------------------------------

    X = df[
        feature_columns
    ].copy()

    # Convert to numeric
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    missing = X.isna().sum()

    missing = missing[
        missing > 0
    ]

    if not missing.empty:

        print(
            "\nMissing values found:"
        )

        print(
            missing
        )

        raise ValueError(
            "Feature dataset contains missing "
            "values."
        )

    # --------------------------------------------------------
    # Check infinite values
    # --------------------------------------------------------

    if not np.isfinite(
        X.to_numpy()
    ).all():

        raise ValueError(
            "Infinite values found in features."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nNumber of samples: {len(X)}"
    )

    print(
        f"Number of features: {X.shape[1]}"
    )

    print(
        f"Number of classes: {y.nunique()}"
    )

    print(
        "\nClasses:"
    )

    print(
        sorted(y.unique())
    )

    print(
        "\nParticipants:"
    )

    print(
        sorted(groups.unique())
    )

    print(
        "\nClass distribution:"
    )

    print(
        y.value_counts()
    )

    return (
        X,
        y,
        groups,
        feature_columns
    )


# ============================================================
# 4. BUILD MODELS
# ============================================================

def build_models():

    models = {

        "Logistic Regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=42
                    )
                )
            ]
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "SVM": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]
        )
    }

    return models


# ============================================================
# 5. PARTICIPANT-WISE CROSS VALIDATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    groups
):

    logo = LeaveOneGroupOut()

    true_labels = []
    predictions = []

    fold_results = []

    fold_number = 1

    # --------------------------------------------------------
    # Every participant becomes test set once
    # --------------------------------------------------------

    for train_index, test_index in logo.split(
        X,
        y,
        groups
    ):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]

        test_group = groups.iloc[
            test_index
        ].iloc[0]

        print(
            f"\nFold {fold_number}"
        )

        print(
            f"Test participant: "
            f"{test_group}"
        )

        print(
            f"Train samples: "
            f"{len(X_train)}"
        )

        print(
            f"Test samples: "
            f"{len(X_test)}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        true_labels.extend(
            y_test.tolist()
        )

        predictions.extend(
            y_pred.tolist()
        )

        fold_accuracy = accuracy_score(
            y_test,
            y_pred
        )

        fold_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        fold_results.append(
            {
                "participant": test_group,
                "accuracy": fold_accuracy,
                "macro_f1": fold_f1
            }
        )

        print(
            f"Accuracy: "
            f"{fold_accuracy:.4f}"
        )

        print(
            f"Macro F1: "
            f"{fold_f1:.4f}"
        )

        fold_number += 1

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall_accuracy = accuracy_score(
        true_labels,
        predictions
    )

    overall_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        true_labels,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=sorted(
            y.unique()
        )
    )

    return {
        "accuracy": overall_accuracy,
        "macro_f1": overall_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "fold_results": fold_results,
        "true_labels": true_labels,
        "predictions": predictions
    }


# ============================================================
# 6. TRAIN AND COMPARE MODELS
# ============================================================

def train_models(
    X,
    y,
    groups
):

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    models = build_models()

    results = {}

    for model_name, model in models.items():

        print(
            "\n" + "=" * 60
        )

        print(
            f"MODEL: {model_name}"
        )

        print(
            "=" * 60
        )

        result = evaluate_model(
            model=model,
            X=X,
            y=y,
            groups=groups
        )

        results[
            model_name
        ] = result

        print(
            "\nOverall accuracy:"
        )

        print(
            f"{result['accuracy']:.4f}"
        )

        print(
            "\nOverall Macro F1:"
        )

        print(
            f"{result['macro_f1']:.4f}"
        )

        print(
            "\nClassification report:"
        )

        print(
            result[
                "classification_report"
            ]
        )

        print(
            "\nConfusion matrix:"
        )

        print(
            np.array(
                result[
                    "confusion_matrix"
                ]
            )
        )

    return models, results


# ============================================================
# 7. SELECT MODEL
# ============================================================

def select_model(
    models,
    results
):

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison = []

    for model_name, result in results.items():

        comparison.append(
            {
                "model": model_name,
                "accuracy": result[
                    "accuracy"
                ],
                "macro_f1": result[
                    "macro_f1"
                ]
            }
        )

    comparison_df = pd.DataFrame(
        comparison
    )

    print(
        "\n"
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Use macro F1 as primary selection metric
    # --------------------------------------------------------

    best_model_name = (
        comparison_df
        .sort_values(
            by=[
                "macro_f1",
                "accuracy"
            ],
            ascending=False
        )
        .iloc[0]["model"]
    )

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    return best_model_name, comparison_df


# ============================================================
# 8. TRAIN FINAL MODEL ON ALL DATA
# ============================================================

def train_final_model(
    model,
    X,
    y
):

    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)

    print(
        "\nTraining on all available "
        "samples..."
    )

    model.fit(
        X,
        y
    )

    print(
        "\nFinal model training completed ✅"
    )

    return model


# ============================================================
# 9. SAVE MODEL + METADATA
# ============================================================

def save_artifacts(
    model,
    model_name,
    feature_columns,
    results,
    comparison_df
):

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR /
        "exercise_classifier.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_file = (
        MODEL_DIR /
        "exercise_features.pkl"
    )

    joblib.dump(
        feature_columns,
        feature_file
    )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes = list(
        model.classes_
    )

    classes_file = (
        MODEL_DIR /
        "exercise_classes.pkl"
    )

    joblib.dump(
        classes,
        classes_file
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    selected_result = results[
        model_name
    ]

    metrics = {

        "selected_model":
            model_name,

        "accuracy":
            float(
                selected_result[
                    "accuracy"
                ]
            ),

        "macro_f1":
            float(
                selected_result[
                    "macro_f1"
                ]
            ),

        "classes":
            classes,

        "feature_count":
            len(feature_columns),

        "fold_results":
            selected_result[
                "fold_results"
            ]
    }

    metrics_file = (
        MODEL_DIR /
        "exercise_metrics.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison_file = (
        MODEL_DIR /
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\nSaved model:"
    )

    print(
        model_file
    )

    print(
        "\nSaved feature list:"
    )

    print(
        feature_file
    )

    print(
        "\nSaved classes:"
    )

    print(
        classes_file
    )

    print(
        "\nSaved metrics:"
    )

    print(
        metrics_file
    )

    print(
        "\nSaved comparison:"
    )

    print(
        comparison_file
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI FITNESS TRACKER")
    print("EXERCISE RECOGNITION MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        X,
        y,
        groups,
        feature_columns
    ) = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Train/evaluate
    # --------------------------------------------------------

    (
        models,
        results
    ) = train_models(
        X=X,
        y=y,
        groups=groups
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    (
        best_model_name,
        comparison_df
    ) = select_model(
        models=models,
        results=results
    )

    # --------------------------------------------------------
    # Final training
    # --------------------------------------------------------

    best_model = models[
        best_model_name
    ]

    final_model = train_final_model(
        model=best_model,
        X=X,
        y=y
    )

    # --------------------------------------------------------
    # Save everything
    # --------------------------------------------------------

    save_artifacts(
        model=final_model,
        model_name=best_model_name,
        feature_columns=feature_columns,
        results=results,
        comparison_df=comparison_df
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXERCISE MODEL PIPELINE COMPLETED ✅")
    print("=" * 60)


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":

    main()