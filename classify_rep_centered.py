import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import periodogram

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_ROOT / "models" / "saved_models" / "exercise_classifier.pkl"
FEATURE_FILE = PROJECT_ROOT / "models" / "saved_models" / "exercise_features.pkl"

ACC_FILE = PROJECT_ROOT / "data" / "processed" / "phone_accelerometer_filtered.csv"
GYRO_FILE = PROJECT_ROOT / "data" / "processed" / "phone_gyroscope_filtered.csv"
REP_FILE = PROJECT_ROOT / "data" / "processed" / "phone_rep_segments.csv"

model = joblib.load(MODEL_FILE)
feature_names = list(joblib.load(FEATURE_FILE))

acc_df = pd.read_csv(ACC_FILE)
gyro_df = pd.read_csv(GYRO_FILE)
rep_df = pd.read_csv(REP_FILE)

acc_time = pd.to_datetime(acc_df["time"])
gyro_time = pd.to_datetime(gyro_df["time"])

acc_start = acc_time.iloc[0]
gyro_start = gyro_time.iloc[0]

acc_df["time_s"] = (
    acc_time - acc_start
).dt.total_seconds()

gyro_df["time_s"] = (
    gyro_time - gyro_start
).dt.total_seconds()

def spectral_features(values, sampling_rate):
    values = np.asarray(values, dtype=float)

    if len(values) < 4:
        return 0.0, 0.0, 0.0

    frequencies, power = periodogram(
        values,
        fs=sampling_rate
    )

    if len(power) == 0 or np.sum(power) <= 0:
        return 0.0, 0.0, 0.0

    power = np.maximum(power, 0.0)

    nonzero = frequencies > 0

    if np.any(nonzero):
        dominant_index = np.argmax(
            power[nonzero]
        )
        dominant_frequency = frequencies[
            nonzero
        ][dominant_index]
    else:
        dominant_frequency = 0.0

    total_power = np.sum(power)

    weighted_frequency = (
        np.sum(frequencies * power)
        / total_power
    )

    probabilities = power / total_power
    probabilities = probabilities[
        probabilities > 0
    ]

    spectral_entropy = (
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
        / np.log2(len(probabilities))
        if len(probabilities) > 1
        else 0.0
    )

    return (
        dominant_frequency,
        weighted_frequency,
        spectral_entropy
    )

def build_features(
    acc_segment,
    gyro_segment
):

    acc_magnitude = np.sqrt(
        acc_segment["ax_filtered"] ** 2
        + acc_segment["ay_filtered"] ** 2
        + acc_segment["az_filtered"] ** 2
    )

    gyro_magnitude = np.sqrt(
        gyro_segment["gx_filtered"] ** 2
        + gyro_segment["gy_filtered"] ** 2
        + gyro_segment["gz_filtered"] ** 2
    )

    features = {}

    features["acc_ax_filtered_mean"] = acc_segment["ax_filtered"].mean()
    features["acc_ax_filtered_std"] = acc_segment["ax_filtered"].std()

    features["acc_ay_filtered_mean"] = acc_segment["ay_filtered"].mean()
    features["acc_ay_filtered_std"] = acc_segment["ay_filtered"].std()

    features["acc_az_filtered_mean"] = acc_segment["az_filtered"].mean()
    features["acc_az_filtered_std"] = acc_segment["az_filtered"].std()

    features["acc_acc_magnitude_filtered_mean"] = acc_magnitude.mean()
    features["acc_acc_magnitude_filtered_std"] = acc_magnitude.std()

    acc_signals = {
        "acc_ax_filtered": acc_segment["ax_filtered"].values,
        "acc_ay_filtered": acc_segment["ay_filtered"].values,
        "acc_az_filtered": acc_segment["az_filtered"].values,
        "acc_acc_magnitude_filtered": acc_magnitude.values
    }

    for name, values in acc_signals.items():
        dominant, weighted, entropy = spectral_features(
            values,
            12.5
        )

        features[f"{name}_dominant_frequency"] = dominant
        features[f"{name}_weighted_frequency"] = weighted
        features[f"{name}_spectral_entropy"] = entropy

    features["acc_acc_magnitude_mean"] = acc_magnitude.mean()
    features["acc_acc_magnitude_std"] = acc_magnitude.std()
    features["acc_acc_magnitude_min"] = acc_magnitude.min()
    features["acc_acc_magnitude_max"] = acc_magnitude.max()

    features["gyro_gx_filtered_mean"] = gyro_segment["gx_filtered"].mean()
    features["gyro_gx_filtered_std"] = gyro_segment["gx_filtered"].std()

    features["gyro_gy_filtered_mean"] = gyro_segment["gy_filtered"].mean()
    features["gyro_gy_filtered_std"] = gyro_segment["gy_filtered"].std()

    features["gyro_gz_filtered_mean"] = gyro_segment["gz_filtered"].mean()
    features["gyro_gz_filtered_std"] = gyro_segment["gz_filtered"].std()

    features["gyro_gyro_magnitude_filtered_mean"] = gyro_magnitude.mean()
    features["gyro_gyro_magnitude_filtered_std"] = gyro_magnitude.std()

    gyro_signals = {
        "gyro_gx_filtered": gyro_segment["gx_filtered"].values,
        "gyro_gy_filtered": gyro_segment["gy_filtered"].values,
        "gyro_gz_filtered": gyro_segment["gz_filtered"].values,
        "gyro_gyro_magnitude_filtered": gyro_magnitude.values
    }

    for name, values in gyro_signals.items():
        dominant, weighted, entropy = spectral_features(
            values,
            25.0
        )

        features[f"{name}_dominant_frequency"] = dominant
        features[f"{name}_weighted_frequency"] = weighted
        features[f"{name}_spectral_entropy"] = entropy

    features["gyro_gyro_magnitude_mean"] = gyro_magnitude.mean()
    features["gyro_gyro_magnitude_std"] = gyro_magnitude.std()
    features["gyro_gyro_magnitude_min"] = gyro_magnitude.min()
    features["gyro_gyro_magnitude_max"] = gyro_magnitude.max()

    return features

results = []

print("=" * 90)
print("REP-CENTERED 4-SECOND CLASSIFICATION")
print("=" * 90)

for _, rep in rep_df.sort_values("rep_id").iterrows():

    rep_id = int(rep["rep_id"])
    center = float(rep["center_time_s"])

    window_start = center - 2.0
    window_end = center + 2.0

    acc_segment = acc_df[
        (acc_df["time_s"] >= window_start)
        & (acc_df["time_s"] <= window_end)
    ].copy()

    gyro_segment = gyro_df[
        (gyro_df["time_s"] >= window_start)
        & (gyro_df["time_s"] <= window_end)
    ].copy()

    if len(acc_segment) < 20 or len(gyro_segment) < 40:
        print()
        print(f"Rep {rep_id}: insufficient samples")
        continue

    feature_dict = build_features(
        acc_segment,
        gyro_segment
    )

    feature_row = pd.DataFrame(
        [[feature_dict[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = model.predict(
        feature_row
    )[0]

    probabilities = model.predict_proba(
        feature_row
    )[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    sorted_probabilities = sorted(
        class_probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results.append({
        "rep": rep_id,
        "center_time": center,
        "prediction": prediction,
        "confidence_percent": probabilities.max() * 100,
        "top_class_1": sorted_probabilities[0][0],
        "top_prob_1": sorted_probabilities[0][1] * 100,
        "top_class_2": sorted_probabilities[1][0],
        "top_prob_2": sorted_probabilities[1][1] * 100,
        "top_class_3": sorted_probabilities[2][0],
        "top_prob_3": sorted_probabilities[2][1] * 100,
    })

result_df = pd.DataFrame(results)

print()

if len(result_df) > 0:
    print("Rep-level predictions:")
    print(
        result_df.to_string(
            index=False,
            formatters={
                "center_time": "{:.2f}".format,
                "confidence_percent": "{:.2f}".format,
                "top_prob_1": "{:.2f}".format,
                "top_prob_2": "{:.2f}".format,
                "top_prob_3": "{:.2f}".format,
            }
        )
    )

    print()
    print("=" * 90)
    print("VALID REPS")
    print("=" * 90)

    valid_df = result_df[
        result_df["rep"] <= 5
    ]

    if len(valid_df) > 0:
        print()
        print("Valid rep prediction distribution:")
        print(
            valid_df["prediction"]
            .value_counts()
            .to_string()
        )

        print()
        print(
            "Average confidence:",
            f"{valid_df['confidence_percent'].mean():.2f}%"
        )

print()
print("=" * 90)
print("REP-CENTERED CLASSIFICATION COMPLETED")
print("=" * 90)
