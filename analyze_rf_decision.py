import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_ROOT / "models" / "saved_models" / "exercise_classifier.pkl"
FEATURE_FILE = PROJECT_ROOT / "models" / "saved_models" / "exercise_features.pkl"
TRAIN_FILE = PROJECT_ROOT / "data" / "features" / "final_feature_dataset.csv"
PHONE_FILE = PROJECT_ROOT / "data" / "processed" / "phone_final_feature_dataset.csv"

model = joblib.load(MODEL_FILE)
model_features = joblib.load(FEATURE_FILE)

train_df = pd.read_csv(TRAIN_FILE)
phone_df = pd.read_csv(PHONE_FILE)

feature_names = list(model_features)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

print()
print("Top 15 features:")
print(
    importance_df.head(15).to_string(
        index=False,
        formatters={"importance": "{:.6f}".format}
    )
)

print()
print("=" * 80)
print("PHONE WINDOW PREDICTIONS")
print("=" * 80)

prediction_probabilities = model.predict_proba(phone_df[feature_names])
predictions = model.predict(phone_df[feature_names])

classes = list(model.classes_)

for index, row in phone_df.iterrows():
    probability_map = {
        classes[class_index]: prediction_probabilities[index][class_index]
        for class_index in range(len(classes))
    }

    print()
    print(f"Window {int(row['window_id'])}")
    print(f"Predicted: {predictions[index]}")

    for class_name, probability in sorted(
        probability_map.items(),
        key=lambda item: item[1],
        reverse=True
    )[:3]:
        print(f"  {class_name}: {probability * 100:.2f}%")

print()
print("=" * 80)
print("TOP FEATURE VALUES BY WINDOW")
print("=" * 80)

top_features = importance_df.head(10)["feature"].tolist()

for _, phone_row in phone_df.iterrows():
    print()
    print(f"Window {int(phone_row['window_id'])}")

    for feature in top_features:
        print(
            f"  {feature}: "
            f"{float(phone_row[feature]):.6f}"
        )

print()
print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 80)
