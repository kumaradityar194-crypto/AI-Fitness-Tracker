from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "features" / "final_feature_dataset.csv"
PHONE_FILE = PROJECT_ROOT / "data" / "processed" / "phone_final_feature_dataset.csv"

MODEL_FEATURES_FILE = (
    PROJECT_ROOT / "models" / "saved_models" / "exercise_features.pkl"
)

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "phone_class_distance.csv"


def load_feature_names():
    import joblib

    feature_names = joblib.load(MODEL_FEATURES_FILE)
    return list(feature_names)


def main():
    print("=" * 78)
    print("AI FITNESS TRACKER - PHONE CLASS DISTANCE ANALYSIS")
    print("=" * 78)

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    phone_df = pd.read_csv(PHONE_FILE)

    feature_names = load_feature_names()

    missing_train = [c for c in feature_names if c not in train_df.columns]
    missing_phone = [c for c in feature_names if c not in phone_df.columns]

    if missing_train:
        raise ValueError(
            f"Missing training features: {missing_train}"
        )

    if missing_phone:
        raise ValueError(
            f"Missing phone features: {missing_phone}"
        )

    if "exercise" not in train_df.columns:
        raise ValueError("Training dataset must contain an exercise column.")

    print(f"Training shape: {train_df.shape}")
    print(f"Phone shape: {phone_df.shape}")
    print(f"Features used: {len(feature_names)}")

    print("\nPreparing feature matrices...")

    train_x = train_df[feature_names].apply(
        pd.to_numeric, errors="coerce"
    )

    phone_x = phone_df[feature_names].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_train = train_x.notna().all(axis=1)
    valid_phone = phone_x.notna().all(axis=1)

    train_x = train_x.loc[valid_train].reset_index(drop=True)
    train_labels = train_df.loc[valid_train, "exercise"].astype(str).reset_index(
        drop=True
    )

    phone_x = phone_x.loc[valid_phone].reset_index(drop=True)

    print(f"Valid training rows: {len(train_x)}")
    print(f"Valid phone rows: {len(phone_x)}")

    classes = sorted(train_labels.unique())

    print("\nClasses:")
    for cls in classes:
        count = int((train_labels == cls).sum())
        print(f"  {cls}: {count} rows")

    print("\nStandardizing features using training statistics...")

    train_mean = train_x.mean()
    train_std = train_x.std(ddof=1).replace(0, 1.0)

    phone_mean = phone_x.mean()

    phone_standardized = (phone_mean - train_mean) / train_std

    results = []

    for cls in classes:
        class_mask = train_labels == cls
        class_x = train_x.loc[class_mask]

        class_mean = class_x.mean()
        class_std = class_x.std(ddof=1).replace(0, 1.0)

        distance = (
            (phone_mean - class_mean) / class_std
        ).replace([np.inf, -np.inf], np.nan).dropna()

        rms_distance = float(
            np.sqrt(np.mean(np.square(distance.to_numpy())))
        )

        median_abs_distance = float(
            np.median(np.abs(distance.to_numpy()))
        )

        results.append(
            {
                "exercise": cls,
                "training_rows": int(class_mask.sum()),
                "rms_standardized_distance": rms_distance,
                "median_absolute_standardized_distance": median_abs_distance,
            }
        )

    result_df = pd.DataFrame(results).sort_values(
        "rms_standardized_distance"
    ).reset_index(drop=True)

    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1))

    print("\n" + "=" * 78)
    print("PHONE DISTANCE TO TRAINING EXERCISE CLASSES")
    print("=" * 78)

    print(result_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("LARGEST GLOBAL PHONE FEATURE SHIFTS")
    print("=" * 78)

    shift_df = pd.DataFrame(
        {
            "feature": feature_names,
            "training_mean": train_mean.values,
            "phone_mean": phone_mean.values,
            "standardized_shift": phone_standardized.values,
            "absolute_shift": np.abs(phone_standardized.values),
        }
    )

    shift_df = shift_df.sort_values(
        "absolute_shift", ascending=False
    ).head(15)

    print(
        shift_df.to_string(
            index=False,
            formatters={
                "training_mean": "{:.6f}".format,
                "phone_mean": "{:.6f}".format,
                "standardized_shift": "{:.6f}".format,
                "absolute_shift": "{:.6f}".format,
            },
        )
    )

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 78)
    print("PHONE CLASS DISTANCE ANALYSIS COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()