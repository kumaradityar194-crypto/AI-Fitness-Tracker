from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_ROOT / "data" / "features" / "final_feature_dataset.csv"
PHONE_FILE = PROJECT_ROOT / "data" / "processed" / "phone_final_feature_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "phone_participant_fold_predictions.csv"

train_df = pd.read_csv(TRAIN_FILE)
phone_df = pd.read_csv(PHONE_FILE)

metadata_columns = {
    "participant",
    "exercise",
    "intensity"
}

feature_columns = [
    column
    for column in train_df.columns
    if column not in metadata_columns
]

train_features = train_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

phone_features = phone_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

valid_train = train_features.notna().all(axis=1)
valid_phone = phone_features.notna().all(axis=1)

train_features = train_features.loc[valid_train].reset_index(drop=True)
train_labels = train_df.loc[
    valid_train, "exercise"
].astype(str).reset_index(drop=True)

train_groups = train_df.loc[
    valid_train, "participant"
].astype(str).reset_index(drop=True)

phone_features = phone_features.loc[
    valid_phone
].reset_index(drop=True)

participants = sorted(train_groups.unique())

results = []

print("=" * 90)
print("CROSS-PARTICIPANT PHONE PREDICTION")
print("=" * 90)

for participant in participants:

    train_mask = train_groups != participant

    fold_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    fold_model.fit(
        train_features.loc[train_mask],
        train_labels.loc[train_mask]
    )

    predictions = fold_model.predict(phone_features)
    probabilities = fold_model.predict_proba(phone_features)

    for index, prediction in enumerate(predictions):

        results.append(
            {
                "held_out_participant": participant,
                "window_id": int(
                    phone_df.loc[
                        phone_features.index[index],
                        "window_id"
                    ]
                ),
                "prediction": prediction,
                "confidence_percent": probabilities[index].max() * 100
            }
        )

result_df = pd.DataFrame(results)

print()
print("Window predictions by held-out participant:")
print(
    result_df.to_string(
        index=False,
        formatters={
            "confidence_percent": "{:.2f}".format
        }
    )
)

print()
print("=" * 90)
print("PREDICTION CONSENSUS")
print("=" * 90)

consensus = (
    result_df
    .groupby("window_id")["prediction"]
    .agg(
        lambda values: values.value_counts().index[0]
    )
)

for window_id, prediction in consensus.items():
    print(
        f"Window {window_id}: {prediction}"
    )

print()
print("Overall prediction counts:")
print(
    result_df["prediction"].value_counts().to_string()
)

print()
print(
    "Mean confidence:",
    f"{result_df['confidence_percent'].mean():.2f}%"
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("=" * 90)
print("CROSS-PARTICIPANT PHONE PREDICTION COMPLETED")
print("=" * 90)
