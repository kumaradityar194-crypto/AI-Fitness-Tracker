import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_ROOT / "data" / "features" / "final_feature_dataset.csv"
REP_SEGMENTS_FILE = PROJECT_ROOT / "data" / "processed" / "phone_rep_segments.csv"
REP_DIR = PROJECT_ROOT / "data" / "processed" / "phone_reps"
GYRO_FILE = PROJECT_ROOT / "data" / "processed" / "phone_gyroscope_filtered.csv"

train_df = pd.read_csv(TRAIN_FILE)
segments_df = pd.read_csv(REP_SEGMENTS_FILE)
gyro_df = pd.read_csv(GYRO_FILE)

feature_names = [
    "acc_ay_filtered_mean",
    "gyro_gz_filtered_std",
    "acc_ax_filtered_mean",
    "acc_az_filtered_mean",
    "gyro_gyro_magnitude_filtered_mean",
    "gyro_gy_filtered_std",
    "acc_ay_filtered_std",
    "acc_acc_magnitude_filtered_std",
    "gyro_gyro_magnitude_filtered_std",
    "acc_acc_magnitude_min",
]

classes = ["squat", "bench", "ohp"]

gyro_df["time"] = pd.to_datetime(
    gyro_df["time"],
    errors="coerce"
)

if gyro_df["time"].isna().any():
    raise ValueError("Invalid gyro timestamps detected.")

first_gyro_time = gyro_df["time"].iloc[0]

gyro_df["time_s"] = (
    gyro_df["time"] - first_gyro_time
).dt.total_seconds()

required_columns = [
    "gx_filtered",
    "gy_filtered",
    "gz_filtered",
    "time_s",
]

missing_columns = [
    column for column in required_columns
    if column not in gyro_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing gyro columns: {missing_columns}"
    )

class_stats = {}

for exercise in classes:

    class_df = train_df[
        train_df["exercise"] == exercise
    ]

    class_stats[exercise] = {}

    for feature in feature_names:

        values = pd.to_numeric(
            class_df[feature],
            errors="coerce"
        ).dropna()

        class_stats[exercise][feature] = {
            "mean": values.mean(),
            "std": values.std()
        }

def calculate_acc_features(rep_df):

    acc_magnitude = np.sqrt(
        rep_df["ax_filtered"] ** 2
        + rep_df["ay_filtered"] ** 2
        + rep_df["az_filtered"] ** 2
    )

    return {
        "acc_ay_filtered_mean": rep_df["ay_filtered"].mean(),
        "acc_ax_filtered_mean": rep_df["ax_filtered"].mean(),
        "acc_az_filtered_mean": rep_df["az_filtered"].mean(),
        "acc_ay_filtered_std": rep_df["ay_filtered"].std(),
        "acc_acc_magnitude_filtered_std": acc_magnitude.std(),
        "acc_acc_magnitude_min": acc_magnitude.min(),
    }

def calculate_gyro_features(start_time, end_time):

    rep_gyro = gyro_df[
        (gyro_df["time_s"] >= start_time)
        & (gyro_df["time_s"] <= end_time)
    ].copy()

    if len(rep_gyro) < 5:
        raise ValueError(
            f"Not enough gyro samples for interval "
            f"{start_time:.3f}-{end_time:.3f}: "
            f"{len(rep_gyro)}"
        )

    gyro_magnitude = np.sqrt(
        rep_gyro["gx_filtered"] ** 2
        + rep_gyro["gy_filtered"] ** 2
        + rep_gyro["gz_filtered"] ** 2
    )

    return {
        "gyro_gz_filtered_std": rep_gyro["gz_filtered"].std(),
        "gyro_gyro_magnitude_filtered_mean": gyro_magnitude.mean(),
        "gyro_gy_filtered_std": rep_gyro["gy_filtered"].std(),
        "gyro_gyro_magnitude_filtered_std": gyro_magnitude.std(),
        "gyro_samples": len(rep_gyro),
    }

results = []

print("=" * 90)
print("25 HZ GYROSCOPE REP-LEVEL DIAGNOSTIC")
print("=" * 90)

print()
print("Gyro samples:", len(gyro_df))
print("Gyro start time:", gyro_df["time"].iloc[0])
print("Gyro end time:", gyro_df["time"].iloc[-1])
print("Gyro duration:", f"{gyro_df['time_s'].iloc[-1]:.2f} sec")

for _, segment in segments_df.sort_values("rep_id").iterrows():

    rep_id = int(segment["rep_id"])

    start_time = float(
        segment["start_time_s"]
    )

    end_time = float(
        segment["end_time_s"]
    )

    rep_file = (
        REP_DIR
        / f"rep_{rep_id:02d}.csv"
    )

    if not rep_file.exists():
        raise FileNotFoundError(
            f"Missing rep file: {rep_file}"
        )

    rep_df = pd.read_csv(rep_file)

    acc_features = calculate_acc_features(
        rep_df
    )

    gyro_features = calculate_gyro_features(
        start_time,
        end_time
    )

    combined_features = {}

    combined_features.update(
        acc_features
    )

    combined_features.update(
        gyro_features
    )

    scores = {}

    for exercise in classes:

        z_scores = []

        for feature in feature_names:

            value = combined_features[feature]

            mean = class_stats[
                exercise
            ][feature]["mean"]

            std = class_stats[
                exercise
            ][feature]["std"]

            if std == 0 or np.isnan(std):
                std = 1.0

            z_scores.append(
                abs(value - mean) / std
            )

        scores[exercise] = np.mean(
            z_scores
        )

    closest_class = min(
        scores,
        key=scores.get
    )

    results.append({
        "rep": rep_id,
        "gyro_samples": gyro_features[
            "gyro_samples"
        ],
        "closest_class": closest_class,
        "squat_distance": scores["squat"],
        "bench_distance": scores["bench"],
        "ohp_distance": scores["ohp"],
    })

result_df = pd.DataFrame(results)

print()
print("=" * 90)
print("REP-LEVEL CLASS DISTANCES")
print("=" * 90)

print(
    result_df.to_string(
        index=False,
        formatters={
            "squat_distance": "{:.4f}".format,
            "bench_distance": "{:.4f}".format,
            "ohp_distance": "{:.4f}".format,
        }
    )
)

print()
print("=" * 90)
print("VALID REP SUMMARY")
print("=" * 90)

valid_df = result_df[
    result_df["rep"] <= 5
].copy()

for exercise in classes:

    print(
        f"{exercise:6s}: "
        f"{valid_df[exercise + '_distance'].mean():.4f}"
    )

print()
print("Closest class across valid reps:")

valid_summary = {
    "squat": valid_df[
        "squat_distance"
    ].mean(),

    "bench": valid_df[
        "bench_distance"
    ].mean(),

    "ohp": valid_df[
        "ohp_distance"
    ].mean(),
}

for exercise, value in sorted(
    valid_summary.items(),
    key=lambda item: item[1]
):

    print(
        f"{exercise:6s}: {value:.4f}"
    )

print()
print("=" * 90)
print("25 HZ GYROSCOPE DIAGNOSTIC COMPLETED")
print("=" * 90)
