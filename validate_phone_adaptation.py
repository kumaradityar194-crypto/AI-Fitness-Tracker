from pathlib import Path

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_adaptation_validation.csv"
)

train_df = pd.read_csv(TRAIN_FILE)
adapt_df = pd.read_csv(ADAPT_FILE)

metadata_columns = {
    "participant",
    "exercise",
    "intensity",
    "phone_session"
}

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

phone_sessions = adapt_df.loc[
    phone_valid,
    "phone_session"
].astype(str).reset_index(drop=True)

sessions = sorted(
    phone_sessions.unique()
)

results = []

print("=" * 90)
print("PHONE ADAPTATION LEAVE-ONE-SESSION-OUT VALIDATION")
print("=" * 90)

print()
print("Base training windows:", len(base_x))
print("Phone adaptation windows:", len(phone_x))
print("Phone sessions:", sessions)

for held_out_session in sessions:

    adaptation_mask = (
        phone_sessions != held_out_session
    )

    adaptation_x = phone_x.loc[
        adaptation_mask
    ].reset_index(drop=True)

    adaptation_y = phone_y.loc[
        adaptation_mask
    ].reset_index(drop=True)

    test_x = phone_x.loc[
        phone_sessions == held_out_session
    ].reset_index(drop=True)

    test_y = phone_y.loc[
        phone_sessions == held_out_session
    ].reset_index(drop=True)

    combined_x = pd.concat(
        [
            base_x,
            adaptation_x
        ],
        ignore_index=True
    )

    combined_y = pd.concat(
        [
            base_y,
            adaptation_y
        ],
        ignore_index=True
    )

    baseline_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    adapted_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    baseline_model.fit(
        base_x,
        base_y
    )

    adapted_model.fit(
        combined_x,
        combined_y
    )

    baseline_predictions = baseline_model.predict(
        test_x
    )

    adapted_predictions = adapted_model.predict(
        test_x
    )

    baseline_accuracy = (
        baseline_predictions == test_y
    ).mean()

    adapted_accuracy = (
        adapted_predictions == test_y
    ).mean()

    baseline_squat_rate = (
        baseline_predictions == "squat"
    ).mean()

    adapted_squat_rate = (
        adapted_predictions == "squat"
    ).mean()

    results.append(
        {
            "held_out_session": held_out_session,
            "test_windows": len(test_x),
            "baseline_accuracy": baseline_accuracy,
            "adapted_accuracy": adapted_accuracy,
            "baseline_squat_rate": baseline_squat_rate,
            "adapted_squat_rate": adapted_squat_rate
        }
    )

    print()
    print("=" * 90)
    print(
        f"HELD-OUT SESSION: "
        f"{held_out_session}"
    )
    print("=" * 90)

    print(
        f"Test windows: {len(test_x)}"
    )

    print(
        f"Baseline squat rate: "
        f"{baseline_squat_rate * 100:.2f}%"
    )

    print(
        f"Adapted squat rate: "
        f"{adapted_squat_rate * 100:.2f}%"
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        f"Adapted accuracy: "
        f"{adapted_accuracy * 100:.2f}%"
    )

result_df = pd.DataFrame(results)

print()
print("=" * 90)
print("OVERALL VALIDATION")
print("=" * 90)

print()

print(
    result_df.to_string(
        index=False,
        formatters={
            "baseline_accuracy": "{:.4f}".format,
            "adapted_accuracy": "{:.4f}".format,
            "baseline_squat_rate": "{:.4f}".format,
            "adapted_squat_rate": "{:.4f}".format
        }
    )
)

print()
print(
    "Mean baseline accuracy:",
    f"{result_df['baseline_accuracy'].mean() * 100:.2f}%"
)

print(
    "Mean adapted accuracy:",
    f"{result_df['adapted_accuracy'].mean() * 100:.2f}%"
)

print(
    "Mean baseline squat rate:",
    f"{result_df['baseline_squat_rate'].mean() * 100:.2f}%"
)

print(
    "Mean adapted squat rate:",
    f"{result_df['adapted_squat_rate'].mean() * 100:.2f}%"
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
print("PHONE ADAPTATION VALIDATION COMPLETED")
print("=" * 90)
