import re
from pathlib import Path
from collections import Counter

import pandas as pd


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. EXPECTED CATEGORIES
# =========================================================

EXERCISES = {
    "bench",
    "dead",
    "deadlift",
    "ohp",
    "row",
    "squat",
    "rest",
}

INTENSITIES = {
    "heavy",
    "medium",
    "sitting",
    "standing",
}


# =========================================================
# 3. HELPER FUNCTION
# =========================================================

def extract_metadata(filename):
    """
    Extract metadata from filenames such as:

    A-bench-heavy2-rpe8_MetaWear_...
    A-squat-medium1-rpe7_MetaWear_...
    A-rest-sitting_MetaWear_...

    Returns:
        participant
        exercise
        intensity
        set_number
        rpe
        original_filename
    """

    name = Path(filename).name

    # -----------------------------------------------------
    # Participant
    # -----------------------------------------------------
    participant_match = re.match(r"^([A-Z])-", name)
    participant = participant_match.group(1) if participant_match else None

    # -----------------------------------------------------
    # Exercise
    # -----------------------------------------------------
    exercise = None

    lower_name = name.lower()

    if "dead" in lower_name:
        exercise = "deadlift"
    elif "bench" in lower_name:
        exercise = "bench"
    elif "ohp" in lower_name:
        exercise = "ohp"
    elif "squat" in lower_name:
        exercise = "squat"
    elif "row" in lower_name:
        exercise = "row"
    elif "rest" in lower_name:
        exercise = "rest"

    # -----------------------------------------------------
    # Intensity / rest state
    # -----------------------------------------------------
    intensity = None

    if "heavy" in lower_name:
        intensity = "heavy"
    elif "medium" in lower_name:
        intensity = "medium"
    elif "sitting" in lower_name:
        intensity = "sitting"
    elif "standing" in lower_name:
        intensity = "standing"

    # -----------------------------------------------------
    # Set number
    #
    # Examples:
    # heavy1
    # heavy2
    # heavy3
    # medium1
    # medium2
    # medium3
    # -----------------------------------------------------
    set_number = None

    set_match = re.search(
        r"(?:heavy|medium)(\d+)",
        lower_name
    )

    if set_match:
        set_number = int(set_match.group(1))

    # -----------------------------------------------------
    # RPE
    #
    # Examples:
    # rpe6
    # rpe7
    # rpe8
    # rpe9
    # -----------------------------------------------------
    rpe = None

    rpe_match = re.search(r"rpe(\d+)", lower_name)

    if rpe_match:
        rpe = int(rpe_match.group(1))

    return {
        "participant": participant,
        "exercise": exercise,
        "intensity": intensity,
        "set_number": set_number,
        "rpe": rpe,
        "original_filename": name,
    }


# =========================================================
# 4. FIND CSV FILES
# =========================================================

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Raw data folder not found:\n{RAW_DIR}"
    )

files = sorted(RAW_DIR.glob("*.csv"))

print("=" * 70)
print("AI FITNESS TRACKER - DATASET METADATA AUDIT")
print("=" * 70)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Raw folder   : {RAW_DIR}")
print(f"CSV files    : {len(files)}")


if not files:
    raise FileNotFoundError(
        "No CSV files found inside data/raw/"
    )


# =========================================================
# 5. EXTRACT METADATA
# =========================================================

records = []

for file in files:
    metadata = extract_metadata(file.name)
    records.append(metadata)


metadata_df = pd.DataFrame(records)


# =========================================================
# 6. DISPLAY SAMPLE METADATA
# =========================================================

print("\n" + "=" * 70)
print("SAMPLE EXTRACTED METADATA")
print("=" * 70)

print(
    metadata_df[
        [
            "participant",
            "exercise",
            "intensity",
            "set_number",
            "rpe",
            "original_filename",
        ]
    ].head(20).to_string(index=False)
)


# =========================================================
# 7. PARTICIPANT SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("PARTICIPANT SUMMARY")
print("=" * 70)

participant_counts = Counter(
    metadata_df["participant"].dropna()
)

for participant, count in sorted(participant_counts.items()):
    print(f"{participant}: {count} CSV files")


# =========================================================
# 8. EXERCISE SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

exercise_counts = Counter(
    metadata_df["exercise"].dropna()
)

for exercise, count in sorted(exercise_counts.items()):
    print(f"{exercise}: {count} CSV files")


# =========================================================
# 9. INTENSITY SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("INTENSITY / STATE SUMMARY")
print("=" * 70)

intensity_counts = Counter(
    metadata_df["intensity"].dropna()
)

for intensity, count in sorted(intensity_counts.items()):
    print(f"{intensity}: {count} CSV files")


# =========================================================
# 10. SET NUMBER SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("SET NUMBER SUMMARY")
print("=" * 70)

set_counts = Counter(
    metadata_df["set_number"].dropna()
)

if set_counts:
    for set_number, count in sorted(set_counts.items()):
        print(f"Set {int(set_number)}: {count} CSV files")
else:
    print("No numbered exercise sets found.")


# =========================================================
# 11. RPE SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("RPE SUMMARY")
print("=" * 70)

rpe_counts = Counter(
    metadata_df["rpe"].dropna()
)

if rpe_counts:
    for rpe, count in sorted(rpe_counts.items()):
        print(f"RPE {int(rpe)}: {count} CSV files")
else:
    print("No RPE values found.")


# =========================================================
# 12. UNKNOWN / UNPARSED FILES
# =========================================================

print("\n" + "=" * 70)
print("FILES WITH UNKNOWN METADATA")
print("=" * 70)

unknown_df = metadata_df[
    metadata_df["participant"].isna()
    | metadata_df["exercise"].isna()
    | metadata_df["intensity"].isna()
]

if unknown_df.empty:
    print("None. All files were successfully parsed.")
else:
    print(
        unknown_df[
            [
                "participant",
                "exercise",
                "intensity",
                "set_number",
                "rpe",
                "original_filename",
            ]
        ].to_string(index=False)
    )


# =========================================================
# 13. POSSIBLE FORM / ERROR TOKENS
# =========================================================

print("\n" + "=" * 70)
print("POSSIBLE FORM / ERROR INFORMATION IN FILENAMES")
print("=" * 70)

keywords = [
    "form",
    "error",
    "incorrect",
    "correct",
    "bad",
    "wrong",
    "fault",
    "mistake",
    "improper",
]

found_tokens = []

for file in files:
    lower_name = file.name.lower()

    for keyword in keywords:
        if keyword in lower_name:
            found_tokens.append(
                (keyword, file.name)
            )


if found_tokens:
    for keyword, filename in found_tokens:
        print(f"{keyword:<12} -> {filename}")
else:
    print(
        "No obvious form/error keywords were found in filenames."
    )


# =========================================================
# 14. SAVE AUDIT CSV
# =========================================================

output_file = OUTPUT_DIR / "dataset_metadata_audit.csv"

metadata_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("AUDIT SAVED")
print("=" * 70)

print(f"Saved to:\n{output_file}")


# =========================================================
# 15. FINAL CONCLUSION
# =========================================================

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

form_columns_possible = [
    col
    for col in metadata_df.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "form",
            "error",
            "label",
            "class",
        ]
    )
]

print("\nMetadata columns created:")
print(list(metadata_df.columns))

print("\nForm/error label columns:")
if form_columns_possible:
    print(form_columns_possible)
else:
    print("None")

print("\nForm/error information in filenames:")
if found_tokens:
    print("Possible form/error tokens found.")
else:
    print("No obvious form/error tokens found.")

print("\nNext step:")
print(
    "Use the audit result to determine whether a supervised "
    "form-detection model can be trained from the current dataset."
)

print("=" * 70)