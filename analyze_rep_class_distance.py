import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_ROOT / "data" / "features" / "final_feature_dataset.csv"
REP_DIR = PROJECT_ROOT / "data" / "processed" / "phone_reps"

train_df = pd.read_csv(TRAIN_FILE)

features = [
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

def lowpass_filter(values, cutoff=1.3, sampling_rate=12.5, order=4):
    nyquist = sampling_rate / 2.0
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype="low")

    if len(values) < max(len(a), len(b)) * 3:
        return np.asarray(values, dtype=float)

    return filtfilt(b, a, np.asarray(values, dtype=float))

def build_rep_features(file_path):
    df = pd.read_csv(file_path)

    for column in ["ax", "ay", "az", "gx", "gy", "gz"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["ax", "ay", "az", "gx", "gy", "gz"]
    ).reset_index(drop=True)

    if "ax_filtered" not in df.columns:
        df["ax_filtered"] = lowpass_filter(df["ax"].values)
        df["ay_filtered"] = lowpass_filter(df["ay"].values)
        df["az_filtered"] = lowpass_filter(df["az"].values)

    gyro_sampling_rate = 12.5

    df["gx_filtered"] = lowpass_filter(
        df["gx"].values,
        sampling_rate=gyro_sampling_rate
    )
    df["gy_filtered"] = lowpass_filter(
        df["gy"].values,
        sampling_rate=gyro_sampling_rate
    )
    df["gz_filtered"] = lowpass_filter(
        df["gz"].values,
        sampling_rate=gyro_sampling_rate
    )

    df["acc_magnitude_filtered"] = np.sqrt(
        df["ax_filtered"] ** 2
        + df["ay_filtered"] ** 2
        + df["az_filtered"] ** 2
    )

    df["gyro_magnitude_filtered"] = np.sqrt(
        df["gx_filtered"] ** 2
        + df["gy_filtered"] ** 2
        + df["gz_filtered"] ** 2
    )

    return {
        "acc_ay_filtered_mean": df["ay_filtered"].mean(),
        "gyro_gz_filtered_std": df["gz_filtered"].std(),
        "acc_ax_filtered_mean": df["ax_filtered"].mean(),
        "acc_az_filtered_mean": df["az_filtered"].mean(),
        "gyro_gyro_magnitude_filtered_mean": df["gyro_magnitude_filtered"].mean(),
        "gyro_gy_filtered_std": df["gy_filtered"].std(),
        "acc_ay_filtered_std": df["ay_filtered"].std(),
        "acc_acc_magnitude_filtered_std": df["acc_magnitude_filtered"].std(),
        "gyro_gyro_magnitude_filtered_std": df["gyro_magnitude_filtered"].std(),
        "acc_acc_magnitude_min": df["acc_magnitude_filtered"].min(),
    }

classes = ["squat", "bench", "ohp"]

class_stats = {}

for exercise in classes:
    class_df = train_df[train_df["exercise"] == exercise]

    class_stats[exercise] = {}

    for feature in features:
        values = pd.to_numeric(
            class_df[feature],
            errors="coerce"
        ).dropna()

        class_stats[exercise][feature] = {
            "mean": values.mean(),
            "std": values.std(),
        }

results = []

rep_files = sorted(REP_DIR.glob("rep_*.csv"))

print("=" * 90)
print("REP-LEVEL DIAGNOSTIC")
print("=" * 90)

for rep_file in rep_files:
    rep_id = rep_file.stem

    rep_features = build_rep_features(rep_file)

    scores = {}

    for exercise in classes:
        z_scores = []

        for feature in features:
            mean = class_stats[exercise][feature]["mean"]
            std = class_stats[exercise][feature]["std"]

            if std == 0 or np.isnan(std):
                std = 1.0

            value = rep_features[feature]
            z_scores.append(abs(value - mean) / std)

        scores[exercise] = np.mean(z_scores)

    closest_class = min(scores, key=scores.get)

    results.append({
        "rep": rep_id,
        "closest_class": closest_class,
        "squat_distance": scores["squat"],
        "bench_distance": scores["bench"],
        "ohp_distance": scores["ohp"],
    })

result_df = pd.DataFrame(results)

print()
print("Rep-level class distances:")
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
print("REP CLASS SUMMARY")
print("=" * 90)

print(
    result_df["closest_class"]
    .value_counts()
    .to_string()
)

print()
print("Closest class across all reps:")

summary = {
    "squat": result_df["squat_distance"].mean(),
    "bench": result_df["bench_distance"].mean(),
    "ohp": result_df["ohp_distance"].mean(),
}

for exercise, value in sorted(
    summary.items(),
    key=lambda item: item[1]
):
    print(f"{exercise:6s}: {value:.4f}")

print()
print("=" * 90)
print("REP-LEVEL DIAGNOSTIC COMPLETED")
print("=" * 90)
