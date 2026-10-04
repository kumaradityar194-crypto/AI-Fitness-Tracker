from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"

REP_SEGMENT_FILE = PROCESSED_DIR / "phone_rep_segments.csv"
ACC_FILE = PROCESSED_DIR / "phone_accelerometer_filtered.csv"
GYRO_FILE = PROCESSED_DIR / "phone_gyroscope_filtered.csv"

MODEL_FILE = SAVED_MODEL_DIR / "exercise_classifier_phone_adapted.pkl"
FEATURE_FILE = SAVED_MODEL_DIR / "exercise_features_phone_adapted.pkl"

OUTPUT_FILE = PROCESSED_DIR / "phone_rep_predictions.csv"
HISTORY_FILE = PROCESSED_DIR / "phone_rep_prediction_history.csv"

ACC_FS = 12.5
GYRO_FS = 25.0
HALF_WINDOW_SEC = 2.0


def calculate_frequency_features(values, sampling_rate):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return 0.0, 0.0, 0.0

    values = values - np.mean(values)

    fft_values = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(
        len(values),
        d=1.0 / sampling_rate,
    )

    power = np.abs(fft_values) ** 2

    if len(power) > 1:
        frequencies = frequencies[1:]
        power = power[1:]

    total_power = np.sum(power)

    if total_power <= 0:
        return 0.0, 0.0, 0.0

    dominant_index = int(np.argmax(power))
    dominant_frequency = float(frequencies[dominant_index])

    weighted_frequency = float(
        np.sum(frequencies * power) / total_power
    )

    probability = power / total_power
    probability = probability[probability > 0]

    spectral_entropy = float(
        -np.sum(probability * np.log2(probability))
    )

    if len(probability) > 1:
        spectral_entropy /= np.log2(len(probability))

    return (
        dominant_frequency,
        weighted_frequency,
        spectral_entropy,
    )


def add_time_features(features, prefix, values, signal_name):
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    features[f"{prefix}_{signal_name}_mean"] = numeric.mean()
    features[f"{prefix}_{signal_name}_std"] = numeric.std()


def add_frequency_features(
    features,
    prefix,
    values,
    signal_name,
    sampling_rate,
):
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    (
        dominant_frequency,
        weighted_frequency,
        spectral_entropy,
    ) = calculate_frequency_features(
        numeric.to_numpy(),
        sampling_rate,
    )

    base_name = (
        signal_name
        if not prefix
        else f"{prefix}_{signal_name}"
    )

    features[
        f"{base_name}_dominant_frequency"
    ] = dominant_frequency

    features[
        f"{base_name}_weighted_frequency"
    ] = weighted_frequency

    features[
        f"{base_name}_spectral_entropy"
    ] = spectral_entropy

def build_rep_feature_row(acc_window, gyro_window):
    features = {}

    acc_window = acc_window.copy()
    gyro_window = gyro_window.copy()

    acc_window["acc_acc_magnitude"] = np.sqrt(
        acc_window["ax"] ** 2
        + acc_window["ay"] ** 2
        + acc_window["az"] ** 2
    )

    acc_window["acc_acc_magnitude_filtered"] = np.sqrt(
        acc_window["ax_filtered"] ** 2
        + acc_window["ay_filtered"] ** 2
        + acc_window["az_filtered"] ** 2
    )

    gyro_window["gyro_gyro_magnitude"] = np.sqrt(
        gyro_window["gx"] ** 2
        + gyro_window["gy"] ** 2
        + gyro_window["gz"] ** 2
    )

    gyro_window["gyro_gyro_magnitude_filtered"] = np.sqrt(
        gyro_window["gx_filtered"] ** 2
        + gyro_window["gy_filtered"] ** 2
        + gyro_window["gz_filtered"] ** 2
    )

    acc_time_signals = [
        ("ax_filtered", "acc_ax_filtered"),
        ("ay_filtered", "acc_ay_filtered"),
        ("az_filtered", "acc_az_filtered"),
        (
            "acc_acc_magnitude_filtered",
            "acc_acc_magnitude_filtered",
        ),
    ]

    for column, feature_prefix in acc_time_signals:
        values = pd.to_numeric(
            acc_window[column],
            errors="coerce",
        ).dropna()

        features[f"{feature_prefix}_mean"] = values.mean()
        features[f"{feature_prefix}_std"] = values.std()

    gyro_time_signals = [
        ("gx_filtered", "gyro_gx_filtered"),
        ("gy_filtered", "gyro_gy_filtered"),
        ("gz_filtered", "gyro_gz_filtered"),
        (
            "gyro_gyro_magnitude_filtered",
            "gyro_gyro_magnitude_filtered",
        ),
    ]

    for column, feature_prefix in gyro_time_signals:
        values = pd.to_numeric(
            gyro_window[column],
            errors="coerce",
        ).dropna()

        features[f"{feature_prefix}_mean"] = values.mean()
        features[f"{feature_prefix}_std"] = values.std()

    acc_frequency_signals = [
        ("ax_filtered", "acc_ax_filtered"),
        ("ay_filtered", "acc_ay_filtered"),
        ("az_filtered", "acc_az_filtered"),
        (
            "acc_acc_magnitude_filtered",
            "acc_acc_magnitude_filtered",
        ),
    ]

    for column, feature_prefix in acc_frequency_signals:
        add_frequency_features(
            features,
            "",
            acc_window[column],
            feature_prefix,
            ACC_FS,
        )

    gyro_frequency_signals = [
        ("gx_filtered", "gyro_gx_filtered"),
        ("gy_filtered", "gyro_gy_filtered"),
        ("gz_filtered", "gyro_gz_filtered"),
        (
            "gyro_gyro_magnitude_filtered",
            "gyro_gyro_magnitude_filtered",
        ),
    ]

    for column, feature_prefix in gyro_frequency_signals:
        add_frequency_features(
            features,
            "",
            gyro_window[column],
            feature_prefix,
            GYRO_FS,
        )

    acc_magnitude_values = pd.to_numeric(
        acc_window["acc_acc_magnitude"],
        errors="coerce",
    ).dropna()

    features["acc_acc_magnitude_mean"] = (
        acc_magnitude_values.mean()
    )
    features["acc_acc_magnitude_std"] = (
        acc_magnitude_values.std()
    )
    features["acc_acc_magnitude_min"] = (
        acc_magnitude_values.min()
    )
    features["acc_acc_magnitude_max"] = (
        acc_magnitude_values.max()
    )

    gyro_magnitude_values = pd.to_numeric(
        gyro_window["gyro_gyro_magnitude"],
        errors="coerce",
    ).dropna()

    features["gyro_gyro_magnitude_mean"] = (
        gyro_magnitude_values.mean()
    )
    features["gyro_gyro_magnitude_std"] = (
        gyro_magnitude_values.std()
    )
    features["gyro_gyro_magnitude_min"] = (
        gyro_magnitude_values.min()
    )
    features["gyro_gyro_magnitude_max"] = (
        gyro_magnitude_values.max()
    )

    return features

def main():
    session_file = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else None
    )

    if session_file is not None:
        session_id = session_file.stem
    else:
        session_id = "current_session"

    if not REP_SEGMENT_FILE.exists():
        raise FileNotFoundError(
            f"Rep segment file not found: {REP_SEGMENT_FILE}"
        )

    if not ACC_FILE.exists():
        raise FileNotFoundError(
            f"Accelerometer file not found: {ACC_FILE}"
        )

    if not GYRO_FILE.exists():
        raise FileNotFoundError(
            f"Gyroscope file not found: {GYRO_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Adapted model not found: {MODEL_FILE}"
        )

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Adapted feature list not found: {FEATURE_FILE}"
        )

    segments = pd.read_csv(REP_SEGMENT_FILE)
    acc = pd.read_csv(ACC_FILE)
    gyro = pd.read_csv(GYRO_FILE)

    for sensor_df in (acc, gyro):
        sensor_df["time"] = pd.to_datetime(
            sensor_df["time"],
            errors="coerce",
        )

        if sensor_df["time"].isna().all():
            raise ValueError(
                "Sensor time column could not be parsed."
            )

    time_origin = min(
        acc["time"].min(),
        gyro["time"].min(),
    )

    acc["time"] = (
        acc["time"] - time_origin
    ).dt.total_seconds()

    gyro["time"] = (
        gyro["time"] - time_origin
    ).dt.total_seconds()

    feature_names = list(
        joblib.load(FEATURE_FILE)
    )

    if "time" not in acc.columns:
        raise ValueError(
            "Accelerometer data does not contain time column."
        )

    if "time" not in gyro.columns:
        raise ValueError(
            "Gyroscope data does not contain time column."
        )

    model = joblib.load(MODEL_FILE)

    results = []

    for _, rep in segments.iterrows():
        rep_id = int(rep["rep_id"])
        center_time = float(rep["center_time_s"])

        window_start = max(
            float(acc["time"].min()),
            center_time - HALF_WINDOW_SEC,
        )

        window_end = min(
            float(acc["time"].max()),
            center_time + HALF_WINDOW_SEC,
        )

        acc_window = acc[
            (acc["time"] >= window_start)
            & (acc["time"] <= window_end)
        ].copy()

        gyro_window = gyro[
            (gyro["time"] >= window_start)
            & (gyro["time"] <= window_end)
        ].copy()

        if len(acc_window) < 10:
            continue

        if len(gyro_window) < 20:
            continue

        feature_row = build_rep_feature_row(
            acc_window,
            gyro_window,
        )

        feature_df = pd.DataFrame(
            [feature_row]
        )

        missing = [
            feature
            for feature in feature_names
            if feature not in feature_df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing model features for rep {rep_id}: {missing}"
            )

        X = feature_df[
            feature_names
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        if X.isna().any().any():
            raise ValueError(
                f"Invalid feature values detected for rep {rep_id}."
            )

        prediction = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            confidence = float(
                np.max(probabilities) * 100.0
            )
        else:
            confidence = np.nan

        results.append(
            {
                "session_id": session_id,
                "rep_id": rep_id,
                "start_time_s": float(
                    rep["start_time_s"]
                ),
                "center_time_s": center_time,
                "end_time_s": float(
                    rep["end_time_s"]
                ),
                "duration_sec": float(
                    rep["duration_sec"]
                ),
                "window_start_s": window_start,
                "window_end_s": window_end,
                "sample_count_acc": len(acc_window),
                "sample_count_gyro": len(gyro_window),
                "predicted_exercise": str(
                    prediction
                ),
                "confidence_percent": round(
                    confidence,
                    2,
                ),
            }
        )

    result_columns = [
        "session_id",
        "rep_id",
        "start_time_s",
        "center_time_s",
        "end_time_s",
        "duration_sec",
        "window_start_s",
        "window_end_s",
        "sample_count_acc",
        "sample_count_gyro",
        "predicted_exercise",
        "confidence_percent",
    ]

    result = pd.DataFrame(
        results,
        columns=result_columns,
    )

    if result.empty:
        result.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        if HISTORY_FILE.exists():
            history = pd.read_csv(
                HISTORY_FILE
            )

            if "session_id" in history.columns:
                history = history[
                    history["session_id"].astype(str)
                    != str(session_id)
                ]
        else:
            history = pd.DataFrame(
                columns=result_columns
            )

        history.to_csv(
            HISTORY_FILE,
            index=False,
        )

        print("=" * 78)
        print("AI FITNESS TRACKER - PER-REP EXERCISE PREDICTION")
        print("=" * 78)

        print(f"\nSession: {session_id}")
        print("Rep predictions generated: 0")
        print("No repetitions detected. Per-rep prediction skipped.")

        print("\nSaved current session:")
        print(OUTPUT_FILE)

        print("\nSaved prediction history:")
        print(HISTORY_FILE)

        return

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    if HISTORY_FILE.exists():
        history = pd.read_csv(
            HISTORY_FILE
        )

        history = history[
            history["session_id"].astype(str)
            != str(session_id)
        ]

        history = pd.concat(
            [history, result],
            ignore_index=True,
        )
    else:
        history = result.copy()

    history.to_csv(
        HISTORY_FILE,
        index=False,
    )

    print("=" * 78)
    print("AI FITNESS TRACKER - PER-REP EXERCISE PREDICTION")
    print("=" * 78)

    print(f"\nSession: {session_id}")
    print(f"Rep predictions generated: {len(result)}")

    print("\nRep-by-rep predictions:")
    print(
        result[
            [
                "rep_id",
                "predicted_exercise",
                "confidence_percent",
                "duration_sec",
            ]
        ].to_string(index=False)
    )

    print("\nPrediction distribution:")
    print(
        result[
            "predicted_exercise"
        ].value_counts().to_string()
    )

    print("\nAverage confidence:")
    print(
        f"{result['confidence_percent'].mean():.2f}%"
    )

    print("\nSaved current session:")
    print(OUTPUT_FILE)

    print("\nSaved prediction history:")
    print(HISTORY_FILE)


if __name__ == "__main__":
    main()
