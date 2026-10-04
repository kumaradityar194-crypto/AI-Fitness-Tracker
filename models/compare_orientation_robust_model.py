from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_feature_dataset.csv"
)

PHONE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_final_feature_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_orientation_robust_predictions.csv"
)


def main():
    print("=" * 78)
    print("AI FITNESS TRACKER - ORIENTATION ROBUST MODEL")
    print("=" * 78)

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    phone_df = pd.read_csv(PHONE_FILE)

    if "exercise" not in train_df.columns:
        raise ValueError("Training dataset must contain an exercise column.")

    feature_names = [
        "acc_acc_magnitude_filtered_mean",
        "acc_acc_magnitude_filtered_std",
        "acc_acc_magnitude_filtered_dominant_frequency",
        "acc_acc_magnitude_filtered_weighted_frequency",
        "acc_acc_magnitude_filtered_spectral_entropy",
        "acc_acc_magnitude_mean",
        "acc_acc_magnitude_std",
        "acc_acc_magnitude_min",
        "acc_acc_magnitude_max",
        "gyro_gyro_magnitude_filtered_mean",
        "gyro_gyro_magnitude_filtered_std",
        "gyro_gyro_magnitude_filtered_dominant_frequency",
        "gyro_gyro_magnitude_filtered_weighted_frequency",
        "gyro_gyro_magnitude_filtered_spectral_entropy",
        "gyro_gyro_magnitude_mean",
        "gyro_gyro_magnitude_std",
        "gyro_gyro_magnitude_min",
        "gyro_gyro_magnitude_max",
    ]

    missing_train = [
        col for col in feature_names
        if col not in train_df.columns
    ]

    missing_phone = [
        col for col in feature_names
        if col not in phone_df.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing training features: {missing_train}"
        )

    if missing_phone:
        raise ValueError(
            f"Missing phone features: {missing_phone}"
        )

    train_x = train_df[feature_names].apply(
        pd.to_numeric,
        errors="coerce"
    )

    phone_x = phone_df[feature_names].apply(
        pd.to_numeric,
        errors="coerce"
    )

    valid_train = train_x.notna().all(axis=1)
    valid_phone = phone_x.notna().all(axis=1)

    train_x = train_x.loc[valid_train].reset_index(drop=True)
    train_y = (
        train_df.loc[valid_train, "exercise"]
        .astype(str)
        .reset_index(drop=True)
    )

    phone_x = phone_x.loc[valid_phone].reset_index(drop=True)

    print(f"Training samples: {len(train_x)}")
    print(f"Phone windows: {len(phone_x)}")
    print(f"Orientation-robust features: {len(feature_names)}")

    print("\nTraining classifier...")

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    groups = train_df.loc[
        valid_train,
        "participant"
    ].astype(str).reset_index(drop=True)

    logo = LeaveOneGroupOut()

    scores = []

    for fold_number, (train_index, test_index) in enumerate(
        logo.split(
            train_x,
            train_y,
            groups
        ),
        start=1
    ):
        fold_model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        fold_model.fit(
            train_x.iloc[train_index],
            train_y.iloc[train_index]
        )

        fold_prediction = fold_model.predict(
            train_x.iloc[test_index]
        )

        fold_accuracy = (
            fold_prediction
            == train_y.iloc[test_index].values
        ).mean()

        test_participant = groups.iloc[
            test_index
        ].iloc[0]

        scores.append(
            {
                "participant": test_participant,
                "accuracy": fold_accuracy
            }
        )

    score_values = [
        item["accuracy"]
        for item in scores
    ]

    print("\n5-FOLD CROSS-VALIDATION")
    print("-" * 50)

    for item in scores:
        print(
            f"Participant {item['participant']}: "
            f"{item['accuracy'] * 100:.2f}%"
        )

    print(
        f"Mean accuracy: "
        f"{sum(score_values) / len(score_values) * 100:.2f}%"
    )

    mean_score = sum(score_values) / len(score_values)

    variance = sum(
        (score - mean_score) ** 2
        for score in score_values
    ) / len(score_values)

    print(
        f"Std deviation: "
        f"{variance ** 0.5 * 100:.2f}%"
    )

    print("\nTraining final model on all training data...")

    model.fit(train_x, train_y)

    phone_predictions = model.predict(phone_x)
    phone_probabilities = model.predict_proba(phone_x)
    classes = list(model.classes_)

    result = pd.DataFrame(
        {
            "window_id": phone_df.loc[
                valid_phone, "window_id"
            ].reset_index(drop=True),
            "predicted_exercise": phone_predictions
        }
    )

    for index, class_name in enumerate(classes):
        result[f"prob_{class_name}"] = phone_probabilities[:, index]

    result["confidence_percent"] = (
        np.max(phone_probabilities, axis=1) * 100.0
    )

    print("\n" + "=" * 78)
    print("PHONE PREDICTIONS")
    print("=" * 78)

    display_columns = [
        "window_id",
        "predicted_exercise"
    ]

    display_columns.extend(
        [f"prob_{class_name}" for class_name in classes]
    )

    display_columns.append("confidence_percent")

    display_df = result[display_columns].copy()

    for class_name in classes:
        display_df[f"prob_{class_name}"] = (
            display_df[f"prob_{class_name}"] * 100
        ).round(2)

    display_df["confidence_percent"] = (
        display_df["confidence_percent"].round(2)
    )

    print(display_df.to_string(index=False))

    print("\nPrediction distribution:")

    distribution = (
        result["predicted_exercise"]
        .value_counts()
    )

    print(distribution.to_string())

    print(
        f"\nAverage confidence: "
        f"{result['confidence_percent'].mean():.2f}%"
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 78)
    print("ORIENTATION ROBUST MODEL COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()