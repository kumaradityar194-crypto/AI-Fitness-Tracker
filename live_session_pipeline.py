from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PHONE_RAW_FILE = RAW_DIR / "phone_squat_fresh.csv"

PIPELINE_SCRIPTS = [
    PROJECT_ROOT / "preprocessing" / "phone_clean.py",
    PROJECT_ROOT / "preprocessing" / "split_phone_sessions.py",
    PROJECT_ROOT / "preprocessing" / "resample_phone.py",
    PROJECT_ROOT / "preprocessing" / "filter_phone.py",
    PROJECT_ROOT / "preprocessing" / "window_phone.py",
    PROJECT_ROOT / "preprocessing" / "magnitude_phone.py",
    PROJECT_ROOT / "preprocessing" / "time_features_phone.py",
    PROJECT_ROOT / "preprocessing" / "frequency_features_phone.py",
    PROJECT_ROOT / "preprocessing" / "combine_phone_features.py",
    PROJECT_ROOT / "models" / "predict_phone_exercise.py",
    PROJECT_ROOT / "models" / "diagnose_phone_predictions.py",
    PROJECT_ROOT / "stable_exercise_prediction.py",
    PROJECT_ROOT / "recognition" / "phone_rep_counter.py",
    PROJECT_ROOT / "recognition" / "phone_rep_segmentation.py",
    PROJECT_ROOT / "analysis" / "phone_form_analysis.py",
    PROJECT_ROOT / "analysis" / "workout_summary.py",
    PROJECT_ROOT / "analysis" / "workout_history.py",
]


def run_script(script_path):
    if not script_path.exists():
        raise FileNotFoundError(
            f"Pipeline script not found: {script_path}"
        )

    print("")
    print("=" * 72)
    print(f"RUNNING: {script_path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)

    env = dict(
        __import__("os").environ
    )

    existing_pythonpath = env.get(
        "PYTHONPATH",
        ""
    )

    project_root_text = str(
        PROJECT_ROOT
    )

    if existing_pythonpath:
        env["PYTHONPATH"] = (
            project_root_text
            + ";"
            + existing_pythonpath
        )
    else:
        env["PYTHONPATH"] = project_root_text

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed with exit code "
            f"{result.returncode}: "
            f"{script_path.name}"
        )


def run_pipeline(live_session_file):
    live_session_file = Path(
        live_session_file
    )

    if not live_session_file.is_absolute():
        live_session_file = (
            PROJECT_ROOT /
            live_session_file
        )

    live_session_file = (
        live_session_file.resolve()
    )

    if not live_session_file.exists():
        raise FileNotFoundError(
            f"Live session file not found: "
            f"{live_session_file}"
        )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_file = None
    had_original_phone_file = (
        PHONE_RAW_FILE.exists()
    )

    if had_original_phone_file:
        backup_file = (
            RAW_DIR /
            "phone_squat_fresh_pipeline_backup.csv"
        )

        shutil.copy2(
            PHONE_RAW_FILE,
            backup_file,
        )

    print("=" * 72)
    print("AI FITNESS TRACKER - LIVE SESSION PIPELINE")
    print("=" * 72)
    print(
        f"Live session: "
        f"{live_session_file}"
    )
    print(
        f"Input size: "
        f"{live_session_file.stat().st_size} bytes"
    )

    try:
        shutil.copy2(
            live_session_file,
            PHONE_RAW_FILE,
        )

        print("")
        print(
            f"Copied live session to: "
            f"{PHONE_RAW_FILE}"
        )

        start_time = time.perf_counter()

        for script in PIPELINE_SCRIPTS:
            run_script(script)

        rep_prediction_script = (
            PROJECT_ROOT
            / "recognition"
            / "phone_rep_predictions.py"
        )

        print("")
        print("=" * 72)
        print("RUNNING: recognition/phone_rep_predictions.py")
        print("=" * 72)

        env = dict(__import__("os").environ)

        existing_pythonpath = env.get(
            "PYTHONPATH",
            ""
        )

        project_root_text = str(
            PROJECT_ROOT
        )

        if existing_pythonpath:
            env["PYTHONPATH"] = (
                project_root_text
                + ";"
                + existing_pythonpath
            )
        else:
            env["PYTHONPATH"] = project_root_text

        result = subprocess.run(
            [
                sys.executable,
                str(rep_prediction_script),
                str(live_session_file),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                "Per-rep prediction failed with exit code "
                f"{result.returncode}: "
                f"{rep_prediction_script.name}"
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print("")
        print("=" * 72)
        print("LIVE SESSION PIPELINE COMPLETED")
        print("=" * 72)
        print(
            f"Total execution time: "
            f"{elapsed:.2f} seconds"
        )

    finally:
        if (
            had_original_phone_file
            and backup_file is not None
        ):
            shutil.copy2(
                backup_file,
                PHONE_RAW_FILE,
            )

            backup_file.unlink(
                missing_ok=True
            )

            print("")
            print(
                "Original phone_squat_fresh.csv restored."
            )

    return True


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python live_session_pipeline.py <live_session_csv>"
        )

    live_session_file = Path(
        sys.argv[1]
    )

    run_pipeline(
        live_session_file
    )


if __name__ == "__main__":
    main()

