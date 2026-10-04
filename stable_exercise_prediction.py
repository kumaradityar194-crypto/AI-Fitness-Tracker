import pandas as pd
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_exercise_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_stable_exercise_prediction.csv"
)

predictions = pd.read_csv(PREDICTION_FILE)

required_columns = [
    "window_id",
    "predicted_exercise",
    "confidence_percent",
]

missing_columns = [
    column
    for column in required_columns
    if column not in predictions.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

predictions = predictions.sort_values(
    "window_id"
).reset_index(drop=True)

raw_labels = predictions[
    "predicted_exercise"
].astype(str).tolist()

raw_confidences = predictions[
    "confidence_percent"
].astype(float).tolist()

stable_labels = []

for index, current_label in enumerate(raw_labels):

    start = max(0, index - 1)
    end = min(len(raw_labels), index + 2)

    neighborhood = raw_labels[start:end]

    counts = Counter(neighborhood)

    most_common = counts.most_common()

    if len(most_common) == 1:
        stable_label = most_common[0][0]

    elif most_common[0][1] > most_common[1][1]:
        stable_label = most_common[0][0]

    else:
        if stable_labels:
            stable_label = stable_labels[-1]
        else:
            stable_label = current_label

    stable_labels.append(stable_label)

predictions["stable_exercise"] = stable_labels

final_counts = Counter(stable_labels)

final_exercise = final_counts.most_common(1)[0][0]

raw_counts = Counter(raw_labels)

raw_support_percent = (
    raw_counts[final_exercise]
    / len(raw_labels)
    * 100
)

stable_support_percent = (
    final_counts[final_exercise]
    / len(stable_labels)
    * 100
)

average_confidence = sum(
    raw_confidences
) / len(raw_confidences)

final_rows = pd.DataFrame([
    {
        "final_exercise": final_exercise,
        "raw_window_count": len(raw_labels),
        "raw_support_percent": raw_support_percent,
        "stable_support_percent": stable_support_percent,
        "average_classifier_confidence_percent": average_confidence,
        "raw_prediction_sequence": " -> ".join(raw_labels),
        "stable_prediction_sequence": " -> ".join(stable_labels),
    }
])

final_rows.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 80)
print("STABLE EXERCISE PREDICTION")
print("=" * 80)

print()
print("Raw prediction sequence:")
print(" -> ".join(raw_labels))

print()
print("Raw prediction distribution:")

for exercise, count in raw_counts.most_common():
    print(
        f"{exercise:10s}: {count}"
    )

print()
print("Stable prediction sequence:")
print(" -> ".join(stable_labels))

print()
print("Stable prediction distribution:")

for exercise, count in final_counts.most_common():
    print(
        f"{exercise:10s}: {count}"
    )

print()
print("=" * 80)
print("FINAL SESSION RESULT")
print("=" * 80)

print(
    "Final exercise                  :",
    final_exercise
)

print(
    "Raw support                     :",
    f"{raw_support_percent:.2f}%"
)

print(
    "Stable support                  :",
    f"{stable_support_percent:.2f}%"
)

print(
    "Average classifier confidence   :",
    f"{average_confidence:.2f}%"
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("=" * 80)
print("STABLE EXERCISE PREDICTION COMPLETED")
print("=" * 80)
