import pandas as pd
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/phone_imu_cleaned.csv"
)

OUTPUT_FILE = Path(
    "data/processed/phone_imu_sessions.csv"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original samples:", len(df))


# --------------------------------------------------
# Calculate time gap
# --------------------------------------------------

gap_seconds = (
    df["timestamp_ms"].diff() / 1000.0
)

# A gap greater than 1 second starts a new session
df["session_id"] = (
    gap_seconds.gt(1.0)
    .cumsum()
)


# --------------------------------------------------
# Session statistics
# --------------------------------------------------

print("\nSessions found:")
print(
    df.groupby("session_id")
      .size()
      .to_string()
)


print("\nSession durations:")

for session_id, group in df.groupby("session_id"):

    duration = (
        group["timestamp_ms"].iloc[-1]
        - group["timestamp_ms"].iloc[0]
    ) / 1000.0

    print(
        f"Session {session_id}: "
        f"{len(group)} samples, "
        f"{duration:.2f} sec"
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)
