import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_ROOT / "models" / "saved_models" / "exercise_classifier.pkl"
FEATURE_FILE = PROJECT_ROOT / "models" / "saved_models" / "exercise_features.pkl"
TRAIN_FILE = PROJECT_ROOT / "data" / "features" / "final_feature_dataset.csv"
PHONE_FILE = PROJECT_ROOT / "data" / "processed" / "phone_final_feature_dataset.csv"

model = joblib.load(MODEL_FILE)
feature_names = list(joblib.load(FEATURE_FILE))

train_df = pd.read_csv(TRAIN_FILE)
phone_df = pd.read_csv(PHONE_FILE)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

top_features = importance_df.head(10)["feature"].tolist()
classes = ["squat", "bench", "ohp"]

print("=" * 90)
print("CLASS-WISE FEATURE DISTRIBUTION")
print("=" * 90)

for feature in top_features:
    print()
    print(f"FEATURE: {feature}")

    for exercise in classes:
        class_values = train_df.loc[
            train_df["exercise"] == exercise,
            feature
        ].astype(float)

        print(
            f"  {exercise:6s} -> "
            f"mean={class_values.mean():.6f}, "
            f"std={class_values.std():.6f}"
        )

print()
print("=" * 90)
print("PHONE WINDOW CLASS DISTANCES FOR TOP FEATURES")
print("=" * 90)

for _, phone_row in phone_df.iterrows():

    print()
    print(f"WINDOW {int(phone_row['window_id'])}")

    class_scores = {}

    for exercise in classes:
        score_sum = 0.0
        count = 0

        for feature in top_features:
            class_values = train_df.loc[
                train_df["exercise"] == exercise,
                feature
            ].astype(float)

            mean = class_values.mean()
            std = class_values.std()

            if std == 0 or np.isnan(std):
                std = 1.0

            value = float(phone_row[feature])

            z = abs(value - mean) / std
            score_sum += z
            count += 1

        class_scores[exercise] = score_sum / count

    for exercise, score in sorted(
        class_scores.items(),
        key=lambda item: item[1]
    ):
        print(f"  {exercise:6s}: {score:.4f}")

    print("  Model prediction:", model.predict(
        phone_df.loc[[phone_row.name], feature_names]
    )[0])

print()
print("=" * 90)
print("CLASS-WISE FEATURE ANALYSIS COMPLETED")
print("=" * 90)
