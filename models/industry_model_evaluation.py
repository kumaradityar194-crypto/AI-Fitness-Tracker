from pathlib import Path
import json
import time
import os

import joblib
import numpy as np
import pandas as pd
import psutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_feature_dataset.csv"
)

ORIGINAL_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "exercise_classifier.pkl"
)

PHONE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "exercise_classifier_phone_adapted.pkl"
)

PHONE_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phone_final_feature_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. LOAD ORIGINAL DATA
# ============================================================

print("=" * 80)
print("INDUSTRY-LEVEL MODEL EVALUATION")
print("=" * 80)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)

required_columns = {
    "participant",
    "exercise",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

metadata_columns = {
    "participant",
    "exercise",
    "intensity",
}

feature_columns = [
    c for c in df.columns
    if c not in metadata_columns
]

X = df[feature_columns].copy()
y = df["exercise"].astype(str)
groups = df["participant"]

X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any():
    raise ValueError("NaN values found in features.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Infinite values found in features.")

classes = sorted(y.unique())

print("Classes:", classes)
print("Features:", len(feature_columns))
print("Samples:", len(X))
print("Groups:", groups.nunique())


# ============================================================
# 2. PARTICIPANT-WISE OOF EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("LEAVE-ONE-GROUP-OUT EVALUATION")
print("=" * 80)

logo = LeaveOneGroupOut()

oof_true = []
oof_pred = []
oof_prob = []

fold_results = []

for fold_id, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    start=1,
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    train_time = time.perf_counter() - start

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    acc = accuracy_score(
        y_test,
        pred,
    )

    f1 = f1_score(
        y_test,
        pred,
        average="macro",
        zero_division=0,
    )

    fold_results.append(
        {
            "fold": fold_id,
            "group": str(
                groups.iloc[test_idx].iloc[0]
            ),
            "test_samples": len(test_idx),
            "accuracy": acc,
            "macro_f1": f1,
            "train_time_sec": train_time,
        }
    )

    oof_true.extend(
        y_test.tolist()
    )

    oof_pred.extend(
        pred.tolist()
    )

    # Align probability columns with global class order.
    fold_prob = np.zeros(
        (len(test_idx), len(classes))
    )

    for j, cls in enumerate(model.classes_):
        cls_idx = classes.index(str(cls))
        fold_prob[:, cls_idx] = prob[:, j]

    oof_prob.append(fold_prob)

    print(
        f"Fold {fold_id}: "
        f"accuracy={acc:.4f}, "
        f"macro_f1={f1:.4f}, "
        f"train={train_time:.4f}s"
    )


oof_true = np.asarray(oof_true)
oof_pred = np.asarray(oof_pred)
oof_prob = np.vstack(oof_prob)


# ============================================================
# 3. CORE METRICS
# ============================================================

accuracy = accuracy_score(
    oof_true,
    oof_pred,
)

macro_precision = precision_score(
    oof_true,
    oof_pred,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    oof_true,
    oof_pred,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    oof_true,
    oof_pred,
    average="macro",
    zero_division=0,
)

weighted_precision = precision_score(
    oof_true,
    oof_pred,
    average="weighted",
    zero_division=0,
)

weighted_recall = recall_score(
    oof_true,
    oof_pred,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    oof_true,
    oof_pred,
    average="weighted",
    zero_division=0,
)


report = classification_report(
    oof_true,
    oof_pred,
    labels=classes,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).T


# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    oof_true,
    oof_pred,
    labels=classes,
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
)

cm_normalized = confusion_matrix(
    oof_true,
    oof_pred,
    labels=classes,
    normalize="true",
)

cm_norm_df = pd.DataFrame(
    cm_normalized,
    index=classes,
    columns=classes,
)


# ============================================================
# 5. ROC-AUC
# ============================================================

y_true_bin = label_binarize(
    oof_true,
    classes=classes,
)

try:
    roc_auc_macro = roc_auc_score(
        y_true_bin,
        oof_prob,
        average="macro",
        multi_class="ovr",
    )

    roc_auc_per_class = {}

    for i, cls in enumerate(classes):
        roc_auc_per_class[cls] = float(
            roc_auc_score(
                y_true_bin[:, i],
                oof_prob[:, i],
            )
        )

except ValueError:
    roc_auc_macro = None
    roc_auc_per_class = {}


# ============================================================
# 6. PR-AUC / AVERAGE PRECISION
# ============================================================

try:
    pr_auc_macro = average_precision_score(
        y_true_bin,
        oof_prob,
        average="macro",
    )

    pr_auc_per_class = {}

    for i, cls in enumerate(classes):
        pr_auc_per_class[cls] = float(
            average_precision_score(
                y_true_bin[:, i],
                oof_prob[:, i],
            )
        )

except ValueError:
    pr_auc_macro = None
    pr_auc_per_class = {}


# ============================================================
# 7. ORIGINAL SAVED MODEL BENCHMARK
# ============================================================

print("\n" + "=" * 80)
print("SAVED MODEL BENCHMARK")
print("=" * 80)

if not ORIGINAL_MODEL_FILE.exists():
    raise FileNotFoundError(
        ORIGINAL_MODEL_FILE
    )

original_model = joblib.load(
    ORIGINAL_MODEL_FILE
)

print(
    "Original model type:",
    type(original_model).__name__,
)

print(
    "Original model size:",
    f"{ORIGINAL_MODEL_FILE.stat().st_size / (1024 ** 2):.3f} MB"
)


# ============================================================
# 8. PHONE-ADAPTED MODEL LATENCY / RESOURCE BENCHMARK
# ============================================================

if not PHONE_MODEL_FILE.exists():
    raise FileNotFoundError(
        PHONE_MODEL_FILE
    )

if not PHONE_FEATURE_FILE.exists():
    raise FileNotFoundError(
        PHONE_FEATURE_FILE
    )

phone_model = joblib.load(
    PHONE_MODEL_FILE
)

phone_df = pd.read_csv(
    PHONE_FEATURE_FILE
)

# Exact feature list stored with phone model.
PHONE_FEATURE_LIST_FILE = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "exercise_features_phone_adapted.pkl"
)

phone_features = joblib.load(
    PHONE_FEATURE_LIST_FILE
)

X_phone = phone_df[
    phone_features
].copy()

X_phone = X_phone.apply(
    pd.to_numeric,
    errors="coerce",
)

if X_phone.isnull().any().any():
    raise ValueError(
        "NaN values found in phone features."
    )

X_phone_np = X_phone.to_numpy(
    dtype=float
)

print(
    "\nPhone model features:",
    X_phone_np.shape[1],
)

print(
    "Phone inference batch:",
    X_phone_np.shape[0],
)

# Warm-up
for _ in range(10):
    phone_model.predict(
        X_phone_np
    )

process = psutil.Process(
    os.getpid()
)

rss_before = (
    process.memory_info().rss
)

process.cpu_percent(
    None
)

iterations = 200

start = time.perf_counter()

peak_rss = rss_before

for _ in range(iterations):

    phone_model.predict(
        X_phone_np
    )

    rss_now = (
        process.memory_info().rss
    )

    peak_rss = max(
        peak_rss,
        rss_now,
    )

elapsed = (
    time.perf_counter()
    - start
)

cpu_percent = (
    process.cpu_percent(
        None
    )
)

avg_batch_latency = (
    elapsed
    / iterations
)

num_windows = X_phone_np.shape[0]

avg_window_latency = (
    avg_batch_latency
    / num_windows
)

throughput = (
    num_windows
    / avg_batch_latency
)

ram_delta_mb = (
    peak_rss
    - rss_before
) / (1024 ** 2)

phone_model_size_mb = (
    PHONE_MODEL_FILE.stat().st_size
    / (1024 ** 2)
)


# ============================================================
# 9. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL INDUSTRY METRICS")
print("=" * 80)

print(
    f"\nAccuracy             : {accuracy * 100:.3f}%"
)

print(
    f"Macro Precision      : {macro_precision * 100:.3f}%"
)

print(
    f"Macro Recall         : {macro_recall * 100:.3f}%"
)

print(
    f"Macro F1             : {macro_f1 * 100:.3f}%"
)

print(
    f"Weighted Precision   : {weighted_precision * 100:.3f}%"
)

print(
    f"Weighted Recall      : {weighted_recall * 100:.3f}%"
)

print(
    f"Weighted F1          : {weighted_f1 * 100:.3f}%"
)

print(
    f"ROC-AUC Macro OvR    : "
    + (
        f"{roc_auc_macro:.4f}"
        if roc_auc_macro is not None
        else "N/A"
    )
)

print(
    f"PR-AUC Macro         : "
    + (
        f"{pr_auc_macro:.4f}"
        if pr_auc_macro is not None
        else "N/A"
    )
)

print("\nPER-CLASS METRICS")
print(report_df[
    ["precision", "recall", "f1-score", "support"]
])

print("\nCONFUSION MATRIX")
print(cm_df)

print("\nNORMALIZED CONFUSION MATRIX")
print(cm_norm_df)

print("\nPER-CLASS ROC-AUC")
for cls, value in roc_auc_per_class.items():
    print(
        f"{cls:10s}: {value:.4f}"
    )

print("\nPER-CLASS PR-AUC")
for cls, value in pr_auc_per_class.items():
    print(
        f"{cls:10s}: {value:.4f}"
    )

print("\nPHONE MODEL BENCHMARK")
print(
    f"Model size             : {phone_model_size_mb:.3f} MB"
)

print(
    f"Average batch latency  : "
    f"{avg_batch_latency * 1000:.3f} ms"
)

print(
    f"Average window latency : "
    f"{avg_window_latency * 1000:.3f} ms"
)

print(
    f"Throughput             : "
    f"{throughput:.2f} windows/sec"
)

print(
    f"Process CPU usage      : "
    f"{cpu_percent:.2f}%"
)

print(
    f"RAM increase           : "
    f"{ram_delta_mb:.3f} MB"
)


# ============================================================
# 10. SAVE CSV / JSON
# ============================================================

report_df.to_csv(
    OUTPUT_DIR
    / "classification_report.csv"
)

cm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix.csv"
)

cm_norm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix_normalized.csv"
)

pd.DataFrame(
    fold_results
).to_csv(
    OUTPUT_DIR
    / "logo_fold_results.csv",
    index=False,
)

metrics = {
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),
    "roc_auc_macro_ovr": roc_auc_macro,
    "pr_auc_macro": pr_auc_macro,
    "roc_auc_per_class": roc_auc_per_class,
    "pr_auc_per_class": pr_auc_per_class,
    "phone_model_size_mb": float(
        phone_model_size_mb
    ),
    "phone_avg_batch_latency_ms": float(
        avg_batch_latency * 1000
    ),
    "phone_avg_window_latency_ms": float(
        avg_window_latency * 1000
    ),
    "phone_throughput_windows_per_sec": float(
        throughput
    ),
    "process_cpu_percent": float(
        cpu_percent
    ),
    "ram_increase_mb": float(
        ram_delta_mb
    ),
}

with open(
    OUTPUT_DIR
    / "industry_metrics.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )

print(
    "\nSaved evaluation files to:"
)

print(
    OUTPUT_DIR
)

print("\nEvaluation completed successfully.")
