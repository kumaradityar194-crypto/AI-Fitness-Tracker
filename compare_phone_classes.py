import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_ROOT / "data" / "features" / "final_feature_dataset.csv"
PHONE_FILE = PROJECT_ROOT / "data" / "processed" / "phone_final_feature_dataset.csv"

train_df = pd.read_csv(TRAIN_FILE)
phone_df = pd.read_csv(PHONE_FILE)

metadata_columns = ["participant", "exercise", "intensity", "window_id"]
feature_columns = [
    column for column in train_df.columns
    if column not in ["participant", "exercise", "intensity"]
]

feature_columns = [column for column in feature_columns if column in phone_df.columns]

classes = ["squat", "bench", "ohp"]

print("=" * 80)
print("PHONE VS TRAINING CLASS DISTRIBUTION")
print("=" * 80)

print()
print("Training samples by class:")
print(train_df["exercise"].value_counts())

results = []

for window_index, phone_row in phone_df.iterrows():
    distances = {}

    for exercise in classes:
        class_df = train_df[train_df["exercise"] == exercise]

        class_mean = class_df[feature_columns].mean()
        class_std = class_df[feature_columns].std().replace(0, 1)

        phone_values = phone_row[feature_columns].astype(float)
        standardized_difference = (phone_values - class_mean) / class_std

        distance = np.sqrt(
            np.mean(standardized_difference.values ** 2)
        )

        distances[exercise] = distance

    closest_class = min(distances, key=distances.get)

    results.append({
        "window_id": int(phone_row["window_id"]),
        "closest_class": closest_class,
        "squat_distance": distances["squat"],
        "bench_distance": distances["bench"],
        "ohp_distance": distances["ohp"],
    })

result_df = pd.DataFrame(results)

print()
print("Window-level class distances:")
print(result_df.to_string(index=False))

print()
print("=" * 80)
print("AVERAGE DISTANCE")
print("=" * 80)

average_distances = {
    "squat": result_df["squat_distance"].mean(),
    "bench": result_df["bench_distance"].mean(),
    "ohp": result_df["ohp_distance"].mean(),
}

for exercise, distance in sorted(
    average_distances.items(),
    key=lambda item: item[1]
):
    print(f"{exercise:10s}: {distance:.4f}")

print()
print("Closest class by average distance:")
print(min(average_distances, key=average_distances.get))

print()
print("=" * 80)
print("COMPARISON COMPLETED")
print("=" * 80)
