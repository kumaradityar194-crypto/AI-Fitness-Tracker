from pathlib import Path
import shutil
import subprocess
import sys
import os
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ADAPT_DIR = PROCESSED_DIR / "phone_adaptation"

PHONE_RAW_FILE = RAW_DIR / "phone_squat_fresh.csv"

LIVE_FILES = [
    RAW_DIR / "live_session_20260930_151457.csv",
    RAW_DIR / "live_session_20260930_152142.csv",
    RAW_DIR / "live_session_20260930_152501.csv",
]

PREPROCESSING_SCRIPTS = [
    PROJECT_ROOT / "preprocessing" / "phone_clean.py",
    PROJECT_ROOT / "preprocessing" / "split_phone_sessions.py",
    PROJECT_ROOT / "preprocessing" / "resample_phone.py",
    PROJECT_ROOT / "preprocessing" / "filter_phone.py",
    PROJECT_ROOT / "preprocessing" / "window_phone.py",
    PROJECT_ROOT / "preprocessing" / "magnitude_phone.py",
    PROJECT_ROOT / "preprocessing" / "time_features_phone.py",
    PROJECT_ROOT / "preprocessing" / "frequency_features_phone.py",
    PROJECT_ROOT / "preprocessing" / "combine_phone_features.py",
]

ADAPT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

backup_file = None

if PHONE_RAW_FILE.exists():
    backup_file = RAW_DIR / "phone_squat_fresh_adaptation_backup.csv"
    shutil.copy2(PHONE_RAW_FILE, backup_file)

env = dict(os.environ)
existing_pythonpath = env.get("PYTHONPATH", "")

if existing_pythonpath:
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        + ";"
        + existing_pythonpath
    )
else:
    env["PYTHONPATH"] = str(PROJECT_ROOT)

try:
    print("=" * 80)
    print("PHONE ADAPTATION DATASET BUILDER")
    print("=" * 80)

    all_features = []

    for live_file in LIVE_FILES:

        if not live_file.exists():
            raise FileNotFoundError(
                f"Live session not found: {live_file}"
            )

        print()
        print("=" * 80)
        print(f"PROCESSING: {live_file.name}")
        print("=" * 80)

        shutil.copy2(
            live_file,
            PHONE_RAW_FILE
        )

        for script in PREPROCESSING_SCRIPTS:

            print()
            print(
                f"Running: "
                f"{script.relative_to(PROJECT_ROOT)}"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(script)
                ],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(result.stderr)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Script failed: {script.name}"
                )

        feature_file = (
            PROCESSED_DIR
            / "phone_final_feature_dataset.csv"
        )

        if not feature_file.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_file}"
            )

        session_features = pd.read_csv(
            feature_file
        )

        session_features["exercise"] = "squat"
        session_features["phone_session"] = (
            live_file.stem
        )

        session_output = (
            ADAPT_DIR
            / f"{live_file.stem}_features.csv"
        )

        session_features.to_csv(
            session_output,
            index=False
        )

        all_features.append(
            session_features
        )

        print()
        print(
            f"Created: {session_output}"
        )

        print(
            f"Windows: {len(session_features)}"
        )

    combined = pd.concat(
        all_features,
        ignore_index=True
    )

    combined_output = (
        ADAPT_DIR
        / "phone_squat_adaptation_dataset.csv"
    )

    combined.to_csv(
        combined_output,
        index=False
    )

    print()
    print("=" * 80)
    print("PHONE ADAPTATION DATASET SUMMARY")
    print("=" * 80)

    print(
        f"Sessions: {combined['phone_session'].nunique()}"
    )

    print(
        f"Total windows: {len(combined)}"
    )

    print(
        f"Exercise labels: "
        f"{combined['exercise'].value_counts().to_dict()}"
    )

    print()
    print("Session window counts:")

    print(
        combined["phone_session"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Combined dataset:")

    print(
        combined_output
    )

finally:

    if backup_file is not None and backup_file.exists():
        shutil.copy2(
            backup_file,
            PHONE_RAW_FILE
        )

        backup_file.unlink(
            missing_ok=True
        )

        print()
        print(
            "Original phone_squat_fresh.csv restored."
        )

print()
print("=" * 80)
print("PHONE ADAPTATION DATASET BUILD COMPLETED")
print("=" * 80)
