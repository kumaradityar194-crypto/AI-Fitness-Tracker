from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_feature_dataset.csv"
)

ADAPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_adaptation"
    / "phone_squat_adaptation_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
)

OUTPUT_MODEL = (
    MODEL_DIR
    / "exercise_classifier_phone_adapted.pkl"
)

OUTPUT_FEATURES = (
    MODEL_DIR
    / "exercise_features_phone_adapted.pkl"
)

train_df = pd.read_csv(TRAIN_FILE)
adapt_df = pd.read_csv(ADAPT_FILE)

feature_columns = [
    column
    for column in train_df.columns
    if column not in {
        "participant",
        "exercise",
        "intensity"
    }
]

feature_columns = [
    column
    for column in feature_columns
    if column in adapt_df.columns
]

base_x = train_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

base_valid = base_x.notna().all(axis=1)

base_x = base_x.loc[
    base_valid
].reset_index(drop=True)

base_y = train_df.loc[
    base_valid,
    "exercise"
].astype(str).reset_index(drop=True)

phone_x = adapt_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

phone_valid = phone_x.notna().all(axis=1)

phone_x = phone_x.loc[
    phone_valid
].reset_index(drop=True)

phone_y = adapt_df.loc[
    phone_valid,
    "exercise"
].astype(str).reset_index(drop=True)

combined_x = pd.concat(
    [
        base_x,
        phone_x
    ],
    ignore_index=True
)

combined_y = pd.concat(
    [
        base_y,
        phone_y
    ],
    ignore_index=True
)

print("=" * 80)
print("PHONE-ADAPTED EXERCISE MODEL")
print("=" * 80)

print()
print("Base training samples:", len(base_x))
print("Phone adaptation samples:", len(phone_x))
print("Combined samples:", len(combined_x))
print("Features:", len(feature_columns))

print()
print("Class distribution after adaptation:")
print(
    combined_y.value_counts().to_string()
)

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train adapted model without sample weighting
model.fit(combined_x, combined_y)

joblib.dump(
    model,
    OUTPUT_MODEL
)

joblib.dump(
    feature_columns,
    OUTPUT_FEATURES
)

print()
print("Model classes:")
print(
    list(model.classes_)
)

print()
print("Saved adapted model:")
print(OUTPUT_MODEL)

print()
print("Saved adapted feature list:")
print(OUTPUT_FEATURES)

print()
print("=" * 80)
print("PHONE-ADAPTED MODEL CREATED")
print("=" * 80)
