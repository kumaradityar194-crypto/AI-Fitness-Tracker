from pathlib import Path
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from scipy.signal import (
    butter,
    filtfilt,
    savgol_filter,
    find_peaks,
)

warnings.filterwarnings("ignore")


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "filtered_accelerometer.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rep_counts.csv"
)

PARTICIPANT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rep_counts_participant_summary.csv"
)


# =========================================================
# 2. GLOBAL SETTINGS
# =========================================================

FS = 12.5

AXES = [
    "ax_filtered",
    "ay_filtered",
    "az_filtered",
]


# =========================================================
# 3. EVALUATION ONLY
# =========================================================
#
# IMPORTANT:
# These values are NEVER used to generate prediction.
#
# They are only used after prediction for evaluation.
# =========================================================

EXPECTED_REPS = {
    "heavy": 5,
    "medium": 10,
    "sitting": 0,
    "standing": 0,
}


# =========================================================
# 4. EXERCISE CONFIGURATION
# =========================================================
#
# V8 keeps V6-style settings for:
#     bench
#     deadlift
#     row
#     squat
#
# OHP uses the improved V7 cadence idea.
#
# Intensity only changes cadence assumptions.
# It does NOT provide the target rep count.
# =========================================================

BASE_CONFIG = {

    "bench": {
        "lowcut": 0.10,
        "highcut": 0.75,

        "min_period_sec": 1.80,
        "max_period_sec": 4.50,

        "preferred_period_sec": 2.60,
        "sigma_sec": 0.75,

        "smooth_sec": 0.70,

        "prominence_std": 0.32,
        "prominence_range": 0.055,

        "distance_factor": 0.88,

        "use_preferred": True,
    },

    "deadlift": {
        "lowcut": 0.08,
        "highcut": 0.65,

        "min_period_sec": 2.10,
        "max_period_sec": 5.00,

        "preferred_period_sec": 3.10,
        "sigma_sec": 0.85,

        "smooth_sec": 0.85,

        "prominence_std": 0.30,
        "prominence_range": 0.055,

        "distance_factor": 0.88,

        "use_preferred": True,
    },

    "ohp": {
        "lowcut": 0.08,
        "highcut": 0.75,

        # V7 improvement
        "min_period_sec": 2.00,
        "max_period_sec": 4.50,

        "preferred_period_sec": 2.80,
        "sigma_sec": 0.70,

        "smooth_sec": 0.85,

        "prominence_std": 0.26,
        "prominence_range": 0.045,

        "distance_factor": 0.86,

        "use_preferred": True,
    },

    "row": {
        "lowcut": 0.12,
        "highcut": 0.80,

        "min_period_sec": 1.40,
        "max_period_sec": 3.50,

        "preferred_period_sec": 1.70,
        "sigma_sec": 0.50,

        "smooth_sec": 0.70,

        "prominence_std": 0.38,
        "prominence_range": 0.065,

        "distance_factor": 0.90,

        "use_preferred": False,
    },

    "squat": {
        "lowcut": 0.08,
        "highcut": 0.70,

        "min_period_sec": 1.70,
        "max_period_sec": 4.50,

        "preferred_period_sec": 2.70,
        "sigma_sec": 0.80,

        "smooth_sec": 0.85,

        "prominence_std": 0.32,
        "prominence_range": 0.055,

        "distance_factor": 0.88,

        "use_preferred": False,
    },
}


# =========================================================
# 5. INTENSITY ADJUSTMENTS
# =========================================================
#
# These are cadence adjustments only.
#
# They do NOT use:
#     expected = 5
#     expected = 10
#
# Heavy:
#     generally allow slower movement cycles.
#
# Medium:
#     allow somewhat faster cadence.
# =========================================================

INTENSITY_ADJUSTMENTS = {

    "heavy": {
        "min_period_factor": 1.05,
        "preferred_factor": 1.05,
        "distance_factor": 1.00,
    },

    "medium": {
        "min_period_factor": 0.92,
        "preferred_factor": 0.95,
        "distance_factor": 0.94,
    },

    "sitting": {
        "min_period_factor": 1.00,
        "preferred_factor": 1.00,
        "distance_factor": 1.00,
    },

    "standing": {
        "min_period_factor": 1.00,
        "preferred_factor": 1.00,
        "distance_factor": 1.00,
    },

    "unknown": {
        "min_period_factor": 1.00,
        "preferred_factor": 1.00,
        "distance_factor": 1.00,
    },
}


# =========================================================
# 6. CANONICAL SOURCE
# =========================================================

def canonical_source_name(
    source_name: str,
) -> str:

    if not isinstance(
        source_name,
        str,
    ):
        return str(source_name)

    source_name = source_name.strip()

    source_name = re.sub(
        r"_1\.4\.41\.csv$",
        "_1.4.4.csv",
        source_name,
    )

    return source_name


# =========================================================
# 7. GET EXERCISE CONFIG
# =========================================================

def get_exercise_config(
    exercise: str,
    intensity: str,
) -> dict:

    config = (
        BASE_CONFIG[
            exercise
        ]
        .copy()
    )

    adjustment = (
        INTENSITY_ADJUSTMENTS.get(
            intensity,
            INTENSITY_ADJUSTMENTS[
                "unknown"
            ],
        )
    )

    config[
        "min_period_sec"
    ] *= adjustment[
        "min_period_factor"
    ]

    config[
        "preferred_period_sec"
    ] *= adjustment[
        "preferred_factor"
    ]

    config[
        "distance_factor"
    ] *= adjustment[
        "distance_factor"
    ]

    return config


# =========================================================
# 8. SAFE SAVGOL
# =========================================================

def safe_savgol(
    signal: np.ndarray,
    fs: float,
    smooth_sec: float,
) -> np.ndarray:

    signal = np.asarray(
        signal,
        dtype=float,
    )

    n = len(signal)

    if n < 7:
        return signal.copy()

    window = int(
        round(
            fs * smooth_sec
        )
    )

    window = max(
        5,
        window,
    )

    if window % 2 == 0:
        window += 1

    if window >= n:
        window = n - 1

    if window % 2 == 0:
        window -= 1

    if window < 5:
        return signal.copy()

    polyorder = min(
        3,
        window - 2,
    )

    try:

        return savgol_filter(
            signal,
            window_length=window,
            polyorder=polyorder,
            mode="interp",
        )

    except Exception:

        return signal.copy()


# =========================================================
# 9. BUTTERWORTH BANDPASS
# =========================================================

def butter_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:

    signal = np.asarray(
        signal,
        dtype=float,
    )

    if len(signal) < 15:
        return signal.copy()

    nyquist = fs / 2.0

    low = max(
        lowcut,
        0.03,
    )

    high = min(
        highcut,
        nyquist * 0.90,
    )

    if low >= high:
        return signal.copy()

    try:

        b, a = butter(
            order,
            [
                low / nyquist,
                high / nyquist,
            ],
            btype="band",
        )

    except Exception:

        return signal.copy()

    required = (
        3
        * max(
            len(a),
            len(b),
        )
    )

    if len(signal) <= required:
        return signal.copy()

    try:

        return filtfilt(
            b,
            a,
            signal,
        )

    except Exception:

        return signal.copy()


# =========================================================
# 10. PREPARE SIGNAL
# =========================================================

def prepare_signal(
    raw_signal: np.ndarray,
    fs: float,
    config: dict,
) -> np.ndarray:

    signal = np.asarray(
        raw_signal,
        dtype=float,
    )

    finite = np.isfinite(
        signal
    )

    if finite.sum() < 30:
        return np.array([])

    # -----------------------------------------------------
    # Interpolate missing data
    # -----------------------------------------------------

    if not finite.all():

        x = np.arange(
            len(signal)
        )

        valid_x = x[
            finite
        ]

        valid_y = signal[
            finite
        ]

        if len(valid_x) < 2:
            return np.array([])

        signal = np.interp(
            x,
            valid_x,
            valid_y,
        )

    # -----------------------------------------------------
    # Remove baseline
    # -----------------------------------------------------

    signal = (
        signal
        - np.median(signal)
    )

    # -----------------------------------------------------
    # Smooth
    # -----------------------------------------------------

    signal = safe_savgol(
        signal,
        fs,
        config[
            "smooth_sec"
        ],
    )

    # -----------------------------------------------------
    # Movement band
    # -----------------------------------------------------

    signal = (
        butter_bandpass_filter(
            signal,
            fs,
            config[
                "lowcut"
            ],
            config[
                "highcut"
            ],
            order=4,
        )
    )

    return signal


# =========================================================
# 11. NORMALIZED AUTOCORRELATION
# =========================================================

def normalized_autocorrelation(
    signal: np.ndarray,
) -> np.ndarray:

    signal = np.asarray(
        signal,
        dtype=float,
    )

    if len(signal) < 10:
        return np.array([])

    signal = (
        signal
        - np.mean(signal)
    )

    denominator = np.sum(
        signal * signal
    )

    if denominator <= 1e-12:
        return np.array([])

    try:

        acf = np.correlate(
            signal,
            signal,
            mode="full",
        )

    except Exception:

        return np.array([])

    acf = acf[
        len(signal) - 1:
    ]

    if len(acf) == 0:
        return np.array([])

    if acf[0] <= 0:
        return np.array([])

    acf = (
        acf
        / acf[0]
    )

    return acf


# =========================================================
# 12. ESTIMATE REP PERIOD
# =========================================================

def estimate_period(
    signal: np.ndarray,
    fs: float,
    config: dict,
) -> tuple[float | None, float]:

    acf = normalized_autocorrelation(
        signal
    )

    if len(acf) == 0:
        return None, 0.0

    min_period = config[
        "min_period_sec"
    ]

    max_period = config[
        "max_period_sec"
    ]

    preferred = config[
        "preferred_period_sec"
    ]

    sigma = config[
        "sigma_sec"
    ]

    min_lag = int(
        round(
            min_period * fs
        )
    )

    max_lag = min(
        int(
            round(
                max_period * fs
            )
        ),
        len(acf) // 2,
    )

    if max_lag <= min_lag:
        return None, 0.0

    lags = np.arange(
        min_lag,
        max_lag + 1,
    )

    correlations = acf[
        lags
    ]

    valid = np.isfinite(
        correlations
    )

    if not valid.any():
        return None, 0.0

    correlations = correlations.copy()

    correlations[
        ~valid
    ] = -1.0

    # -----------------------------------------------------
    # Local peaks
    # -----------------------------------------------------

    local_peaks, _ = find_peaks(
        correlations,
        distance=max(
            2,
            int(
                0.50 * fs
            ),
        ),
        prominence=0.01,
    )

    candidates = []

    # Strongest correlation
    candidates.append(
        int(
            np.argmax(
                correlations
            )
        )
    )

    for p in local_peaks:

        candidates.append(
            int(p)
        )

    # Top few
    top_n = min(
        12,
        len(correlations),
    )

    top_indices = np.argsort(
        correlations
    )[
        -top_n:
    ]

    for p in top_indices:

        candidates.append(
            int(p)
        )

    candidates = list(
        dict.fromkeys(
            candidates
        )
    )

    best_period = None
    best_score = -np.inf

    for idx in candidates:

        if (
            idx < 0
            or idx >= len(
                correlations
            )
        ):
            continue

        corr = float(
            correlations[
                idx
            ]
        )

        if corr <= 0:
            continue

        period = (
            lags[idx]
            / fs
        )

        # -------------------------------------------------
        # Cadence prior
        # -------------------------------------------------

        if config[
            "use_preferred"
        ]:

            prior = np.exp(
                -0.5
                * (
                    (
                        period
                        - preferred
                    )
                    / max(
                        sigma,
                        1e-8,
                    )
                )
                ** 2
            )

        else:

            prior = 1.0

        # -------------------------------------------------
        # Boundary handling
        # -------------------------------------------------

        boundary_factor = 1.0

        if (
            period
            <= min_period * 1.05
        ):

            boundary_factor *= 0.88

        if (
            period
            >= max_period * 0.95
        ):

            boundary_factor *= 0.92

        score = (
            0.55
            * corr
            + 0.35
            * prior
            + 0.10
            * boundary_factor
        )

        if score > best_score:

            best_score = score

            best_period = float(
                period
            )

    if best_period is None:
        return None, 0.0

    return (
        best_period,
        float(
            best_score
        ),
    )


# =========================================================
# 13. FUNDAMENTAL FILTER
# =========================================================

def fundamental_filter(
    signal: np.ndarray,
    fs: float,
    period: float,
) -> np.ndarray:

    if (
        period is None
        or period <= 0
    ):
        return signal.copy()

    frequency = (
        1.0
        / period
    )

    lowcut = max(
        0.055,
        0.70 * frequency,
    )

    highcut = min(
        0.95,
        1.30 * frequency,
    )

    if highcut <= lowcut:
        return signal.copy()

    return (
        butter_bandpass_filter(
            signal,
            fs,
            lowcut,
            highcut,
            order=3,
        )
    )


# =========================================================
# 14. PEAK DETECTION
# =========================================================

def detect_peaks(
    signal: np.ndarray,
    fs: float,
    period: float,
    config: dict,
    prominence_multiplier: float,
    distance_multiplier: float,
):

    std = float(
        np.std(signal)
    )

    signal_range = float(
        np.ptp(signal)
    )

    if std < 1e-8:

        return (
            np.array(
                [],
                dtype=int,
            ),
            "none",
            0.0,
        )

    prominence = max(
        config[
            "prominence_std"
        ] * std,
        config[
            "prominence_range"
        ] * signal_range,
    )

    prominence *= (
        prominence_multiplier
    )

    distance_sec = max(
        config[
            "min_period_sec"
        ],
        (
            config[
                "distance_factor"
            ]
            * distance_multiplier
            * period
        ),
    )

    distance_samples = max(
        2,
        int(
            round(
                distance_sec * fs
            )
        ),
    )

    # -----------------------------------------------------
    # Positive
    # -----------------------------------------------------

    positive_peaks, positive_props = (
        find_peaks(
            signal,
            distance=distance_samples,
            prominence=prominence,
        )
    )

    # -----------------------------------------------------
    # Negative
    # -----------------------------------------------------

    negative_peaks, negative_props = (
        find_peaks(
            -signal,
            distance=distance_samples,
            prominence=prominence,
        )
    )

    # -----------------------------------------------------
    # Score direction
    # -----------------------------------------------------

    def score_direction(
        peaks,
        props,
    ):

        if len(peaks) == 0:
            return 0.0

        if len(peaks) == 1:
            return 0.10

        intervals = (
            np.diff(peaks)
            / fs
        )

        mean_interval = float(
            np.mean(
                intervals
            )
        )

        if mean_interval <= 0:
            return 0.0

        cv = (
            np.std(
                intervals
            )
            / mean_interval
        )

        period_error = (
            abs(
                mean_interval
                - period
            )
            / max(
                period,
                1e-8,
            )
        )

        regularity = np.exp(
            -cv
        )

        period_fit = np.exp(
            -period_error
        )

        prom_values = (
            props.get(
                "prominences",
                np.array(
                    [0.0]
                ),
            )
        )

        prom_quality = min(
            np.mean(
                prom_values
            )
            / max(
                prominence,
                1e-8,
            ),
            3.0,
        ) / 3.0

        score = (
            0.45
            * regularity
            + 0.35
            * period_fit
            + 0.20
            * prom_quality
        )

        return float(
            score
        )

    positive_score = score_direction(
        positive_peaks,
        positive_props,
    )

    negative_score = score_direction(
        negative_peaks,
        negative_props,
    )

    if (
        positive_score
        >= negative_score
    ):

        return (
            positive_peaks,
            "positive_peak",
            positive_score,
        )

    return (
        negative_peaks,
        "negative_peak",
        negative_score,
    )


# =========================================================
# 15. CLEAN PEAKS
# =========================================================

def clean_peaks(
    peaks: np.ndarray,
    fs: float,
    period: float,
    config: dict,
) -> np.ndarray:

    if len(peaks) <= 1:
        return peaks

    cleaned = [
        int(
            peaks[0]
        )
    ]

    min_allowed = max(
        config[
            "min_period_sec"
        ]
        * 0.80,

        period * 0.70,
    )

    for peak in peaks[1:]:

        gap = (
            peak
            - cleaned[-1]
        ) / fs

        if gap >= min_allowed:

            cleaned.append(
                int(
                    peak
                )
            )

    return np.asarray(
        cleaned,
        dtype=int,
    )


# =========================================================
# 16. REMOVE BOUNDARY PEAKS
# =========================================================

def remove_boundary_peaks(
    peaks: np.ndarray,
    signal_length: int,
    fs: float,
    period: float,
) -> np.ndarray:

    if len(peaks) <= 2:
        return peaks

    boundary_samples = max(
        1,
        int(
            0.15
            * period
            * fs
        ),
    )

    valid = peaks[
        (peaks >= boundary_samples)
        &
        (
            peaks
            < (
                signal_length
                - boundary_samples
            )
        )
    ]

    if len(valid) >= len(peaks) - 1:

        return valid

    return peaks


# =========================================================
# 17. PEAK STABILITY
# =========================================================
#
# Run detector with several nearby parameter values.
#
# Example:
#
# prominence:
#     0.90x
#     1.00x
#     1.10x
#
# distance:
#     0.94x
#     1.00x
#     1.06x
#
# Stable count = better candidate.
# =========================================================

def stability_analysis(
    signal: np.ndarray,
    fs: float,
    period: float,
    config: dict,
):

    variants = [

        (
            0.90,
            0.94,
        ),

        (
            0.95,
            0.97,
        ),

        (
            1.00,
            1.00,
        ),

        (
            1.05,
            1.03,
        ),

        (
            1.10,
            1.06,
        ),
    ]

    candidates = []

    for prominence_multiplier, distance_multiplier in variants:

        peaks, direction, score = (
            detect_peaks(
                signal,
                fs,
                period,
                config,
                prominence_multiplier,
                distance_multiplier,
            )
        )

        peaks = clean_peaks(
            peaks,
            fs,
            period,
            config,
        )

        peaks = remove_boundary_peaks(
            peaks,
            len(signal),
            fs,
            period,
        )

        count = int(
            len(peaks)
        )

        # -------------------------------------------------
        # Timing quality
        # -------------------------------------------------

        if len(peaks) >= 3:

            intervals = (
                np.diff(peaks)
                / fs
            )

            mean_interval = float(
                np.mean(
                    intervals
                )
            )

            if mean_interval > 0:

                cv = (
                    np.std(
                        intervals
                    )
                    / mean_interval
                )

                period_error = (
                    abs(
                        mean_interval
                        - period
                    )
                    / max(
                        period,
                        1e-8,
                    )
                )

                timing_quality = (
                    np.exp(
                        -cv
                    )
                    * np.exp(
                        -period_error
                    )
                )

            else:

                timing_quality = 0.0

        elif len(peaks) == 2:

            interval = (
                abs(
                    peaks[1]
                    - peaks[0]
                )
                / fs
            )

            period_error = (
                abs(
                    interval
                    - period
                )
                / max(
                    period,
                    1e-8,
                )
            )

            timing_quality = np.exp(
                -period_error
            )

        else:

            timing_quality = 0.0

        final_score = (
            0.65
            * score
            + 0.35
            * timing_quality
        )

        candidates.append(
            {
                "count": count,
                "direction": direction,
                "score": float(
                    final_score
                ),
                "prominence_multiplier":
                    prominence_multiplier,
                "distance_multiplier":
                    distance_multiplier,
            }
        )

    # -----------------------------------------------------
    # Remove zero-count candidates
    # -----------------------------------------------------

    valid = [
        c
        for c in candidates
        if c["count"] > 0
    ]

    if len(valid) == 0:

        return {
            "final_count": 0,
            "stability_score": 0.0,
            "method": "none",
            "candidate_counts": [],
            "candidate_scores": [],
        }

    # -----------------------------------------------------
    # Weighted vote
    # -----------------------------------------------------

    count_scores = {}

    for candidate in valid:

        count = candidate[
            "count"
        ]

        score = candidate[
            "score"
        ]

        if count not in count_scores:

            count_scores[
                count
            ] = 0.0

        count_scores[
            count
        ] += score

    # -----------------------------------------------------
    # Consensus
    # -----------------------------------------------------

    count_frequency = Counter(
        [
            c["count"]
            for c in valid
        ]
    )

    best_count = None
    best_total_score = -np.inf

    for count, total_score in (
        count_scores.items()
    ):

        frequency = (
            count_frequency[
                count
            ]
        )

        consensus_bonus = (
            0.10
            * frequency
        )

        total = (
            total_score
            + consensus_bonus
        )

        if total > best_total_score:

            best_total_score = total

            best_count = count

    # -----------------------------------------------------
    # Stability
    # -----------------------------------------------------

    selected_frequency = (
        count_frequency[
            best_count
        ]
    )

    stability_score = (
        selected_frequency
        / len(valid)
    )

    # Average quality of selected candidates
    selected_scores = [
        c["score"]
        for c in valid
        if c["count"]
        == best_count
    ]

    if len(
        selected_scores
    ) > 0:

        average_quality = float(
            np.mean(
                selected_scores
            )
        )

    else:

        average_quality = 0.0

    final_stability = (
        0.65
        * stability_score
        + 0.35
        * average_quality
    )

    # -----------------------------------------------------
    # Method
    # -----------------------------------------------------

    if selected_frequency >= 4:

        method = "stable"

    elif selected_frequency == 3:

        method = "mostly_stable"

    elif selected_frequency == 2:

        method = "weak_consensus"

    else:

        method = "unstable"

    return {
        "final_count": int(
            best_count
        ),

        "stability_score":
            float(
                final_stability
            ),

        "method":
            method,

        "candidate_counts":
            [
                c["count"]
                for c in candidates
            ],

        "candidate_scores":
            [
                round(
                    c["score"],
                    4,
                )
                for c in candidates
            ],
    }


# =========================================================
# 18. AXIS QUALITY
# =========================================================

def calculate_axis_quality(
    signal: np.ndarray,
    period: float,
    period_score: float,
    detected_reps: int,
    stability_score: float,
    fs: float,
) -> float:

    if len(signal) < 10:
        return 0.0

    std = float(
        np.std(signal)
    )

    amplitude = float(
        np.ptp(signal)
    )

    if std <= 1e-8:
        return 0.0

    amplitude_ratio = (
        amplitude
        / std
    )

    amplitude_quality = min(
        amplitude_ratio / 8.0,
        1.0,
    )

    # -----------------------------------------------------
    # Rep spacing quality
    # -----------------------------------------------------

    # A weak count with very few reps should not dominate
    # a stable axis.
    count_quality = min(
        detected_reps / 8.0,
        1.0,
    )

    quality = (
        0.40
        * period_score
        + 0.35
        * stability_score
        + 0.15
        * amplitude_quality
        + 0.10
        * count_quality
    )

    return float(
        quality
    )


# =========================================================
# 19. PROCESS ONE AXIS
# =========================================================

def process_axis(
    df: pd.DataFrame,
    axis: str,
    exercise: str,
    intensity: str,
    fs: float,
):

    if axis not in df.columns:
        return None

    config = get_exercise_config(
        exercise,
        intensity,
    )

    raw = pd.to_numeric(
        df[
            axis
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float,
    )

    if np.isfinite(
        raw
    ).sum() < 40:

        return None

    # -----------------------------------------------------
    # Prepare
    # -----------------------------------------------------

    broad_signal = prepare_signal(
        raw,
        fs,
        config,
    )

    if len(broad_signal) < int(
        fs * 5
    ):

        return None

    # -----------------------------------------------------
    # Period
    # -----------------------------------------------------

    period, period_score = (
        estimate_period(
            broad_signal,
            fs,
            config,
        )
    )

    if period is None:
        return None

    # -----------------------------------------------------
    # Fundamental signal
    # -----------------------------------------------------

    signal = fundamental_filter(
        broad_signal,
        fs,
        period,
    )

    signal = safe_savgol(
        signal,
        fs,
        max(
            0.40,
            config[
                "smooth_sec"
            ] * 0.55,
        ),
    )

    # -----------------------------------------------------
    # Stability analysis
    # -----------------------------------------------------

    stability = stability_analysis(
        signal,
        fs,
        period,
        config,
    )

    detected_reps = (
        stability[
            "final_count"
        ]
    )

    stability_score = (
        stability[
            "stability_score"
        ]
    )

    # -----------------------------------------------------
    # Axis quality
    # -----------------------------------------------------

    axis_quality = (
        calculate_axis_quality(
            signal,
            period,
            period_score,
            detected_reps,
            stability_score,
            fs,
        )
    )

    return {

        "axis":
            axis,

        "period":
            float(
                period
            ),

        "frequency":
            float(
                1.0
                / period
            ),

        "period_score":
            float(
                period_score
            ),

        "detected_reps":
            int(
                detected_reps
            ),

        "stability_score":
            float(
                stability_score
            ),

        "axis_quality":
            float(
                axis_quality
            ),

        "method":
            stability[
                "method"
            ],

        "candidate_counts":
            stability[
                "candidate_counts"
            ],

        "candidate_scores":
            stability[
                "candidate_scores"
            ],
    }


# =========================================================
# 20. CHOOSE BEST AXIS
# =========================================================

def choose_best_axis(
    df: pd.DataFrame,
    exercise: str,
    intensity: str,
    fs: float,
):

    if exercise not in BASE_CONFIG:
        return None

    candidates = []

    for axis in AXES:

        result = process_axis(
            df,
            axis,
            exercise,
            intensity,
            fs,
        )

        if result is None:
            continue

        candidates.append(
            result
        )

    if len(candidates) == 0:
        return None

    # -----------------------------------------------------
    # Sort by quality
    # -----------------------------------------------------

    candidates.sort(
        key=lambda x: x[
            "axis_quality"
        ],
        reverse=True,
    )

    best = candidates[0]

    # -----------------------------------------------------
    # OHP special protection
    # -----------------------------------------------------
    #
    # For OHP we prefer an axis with a more realistic
    # cadence when quality is close.
    #
    # No expected rep count is used.
    # -----------------------------------------------------

    if exercise == "ohp":

        reasonable = [
            c
            for c in candidates
            if 2.00
            <= c[
                "period"
            ]
            <= 4.20
        ]

        if len(reasonable) > 0:

            reasonable.sort(
                key=lambda x: x[
                    "axis_quality"
                ],
                reverse=True,
            )

            alternative = (
                reasonable[0]
            )

            if (
                alternative[
                    "axis_quality"
                ]
                >=
                best[
                    "axis_quality"
                ]
                * 0.94
            ):

                best = alternative

    return best


# =========================================================
# 21. PROCESS RECORDING
# =========================================================

def count_repetitions_for_recording(
    df: pd.DataFrame,
    fs: float,
):

    result = {

        "selected_axis":
            "none",

        "estimated_period_sec":
            np.nan,

        "estimated_frequency_hz":
            np.nan,

        "period_score":
            0.0,

        "stability_score":
            0.0,

        "axis_quality":
            0.0,

        "detected_reps":
            0,

        "counting_method":
            "none",

        "candidate_counts":
            "",

        "candidate_scores":
            "",
    }

    exercise = str(
        df[
            "exercise"
        ].iloc[0]
    ).lower().strip()

    intensity = str(
        df[
            "intensity"
        ].iloc[0]
    ).lower().strip()

    # -----------------------------------------------------
    # REST
    # -----------------------------------------------------

    if exercise == "rest":

        return result

    # -----------------------------------------------------
    # Unsupported
    # -----------------------------------------------------

    if exercise not in BASE_CONFIG:

        return result

    # -----------------------------------------------------
    # Best axis
    # -----------------------------------------------------

    best = choose_best_axis(
        df,
        exercise,
        intensity,
        fs,
    )

    if best is None:

        return result

    result[
        "selected_axis"
    ] = best[
        "axis"
    ]

    result[
        "estimated_period_sec"
    ] = round(
        best[
            "period"
        ],
        3,
    )

    result[
        "estimated_frequency_hz"
    ] = round(
        best[
            "frequency"
        ],
        4,
    )

    result[
        "period_score"
    ] = round(
        best[
            "period_score"
        ],
        4,
    )

    result[
        "stability_score"
    ] = round(
        best[
            "stability_score"
        ],
        4,
    )

    result[
        "axis_quality"
    ] = round(
        best[
            "axis_quality"
        ],
        4,
    )

    result[
        "detected_reps"
    ] = int(
        best[
            "detected_reps"
        ]
    )

    result[
        "counting_method"
    ] = best[
        "method"
    ]

    result[
        "candidate_counts"
    ] = ",".join(
        str(x)
        for x in best[
            "candidate_counts"
        ]
    )

    result[
        "candidate_scores"
    ] = ",".join(
        f"{x:.4f}"
        for x in best[
            "candidate_scores"
        ]
    )

    return result


# =========================================================
# 22. PARTICIPANT SUMMARY
# =========================================================

def create_participant_summary(
    results_df: pd.DataFrame,
):

    valid = results_df[
        results_df[
            "expected_reps"
        ].notna()
    ].copy()

    if len(valid) == 0:

        return pd.DataFrame()

    rows = []

    for participant in sorted(
        valid[
            "participant"
        ]
        .dropna()
        .unique()
    ):

        temp = valid[
            valid[
                "participant"
            ]
            == participant
        ]

        mae = float(
            temp[
                "absolute_error"
            ].mean()
        )

        exact = float(
            temp[
                "absolute_error"
            ]
            .eq(0)
            .mean()
            * 100
        )

        rows.append(
            {

                "participant":
                    participant,

                "recordings":
                    len(temp),

                "mae":
                    round(
                        mae,
                        3,
                    ),

                "exact_match_pct":
                    round(
                        exact,
                        2,
                    ),

                "mean_detected_reps":
                    round(
                        temp[
                            "detected_reps"
                        ].mean(),
                        3,
                    ),

                "mean_expected_reps":
                    round(
                        temp[
                            "expected_reps"
                        ].mean(),
                        3,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# 23. MAIN
# =========================================================

def main():

    print()
    print("=" * 85)
    print(
        "AI FITNESS TRACKER - REP COUNTER V8"
    )
    print("=" * 85)
    print()

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Input file   : {INPUT_FILE}"
    )

    print()

    # =====================================================
    # LOAD
    # =====================================================

    if not INPUT_FILE.exists():

        print("ERROR ❌")
        print()

        print(
            "Filtered accelerometer file not found:"
        )

        print(
            INPUT_FILE
        )

        return

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Loaded accelerometer data: "
        f"{df.shape[0]:,} rows x "
        f"{df.shape[1]} columns"
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    required_columns = [
        "source_file",
        "exercise",
        "intensity",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        print()
        print("ERROR ❌")

        print(
            "Missing columns:"
        )

        print(
            missing
        )

        return

    # =====================================================
    # CANONICAL SOURCE
    # =====================================================

    df[
        "canonical_source"
    ] = (
        df[
            "source_file"
        ]
        .astype(str)
        .apply(
            canonical_source_name
        )
    )

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    unique_sources = (
        df[
            [
                "source_file",
                "canonical_source",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "source_file"
        )
    )

    selected_sources = (
        unique_sources
        .drop_duplicates(
            subset=[
                "canonical_source"
            ],
            keep="first",
        )
    )

    selected_source_set = set(
        selected_sources[
            "source_file"
        ]
    )

    before_rows = len(
        df
    )

    df = df[
        df[
            "source_file"
        ].isin(
            selected_source_set
        )
    ].copy()

    after_rows = len(
        df
    )

    duplicate_groups = (
        unique_sources[
            "canonical_source"
        ]
        .duplicated()
        .sum()
    )

    print()
    print(
        "Duplicate handling"
    )

    print(
        "-" * 85
    )

    print(
        "Unique source files before dedupe : "
        f"{len(unique_sources)}"
    )

    print(
        "Duplicate recording groups removed: "
        f"{duplicate_groups}"
    )

    print(
        "Rows retained                     : "
        f"{after_rows:,} / "
        f"{before_rows:,}"
    )

    # =====================================================
    # GROUP
    # =====================================================

    groups = df.groupby(
        "source_file",
        sort=False,
    )

    total_recordings = len(
        groups
    )

    print()

    print(
        f"Recordings to process: "
        f"{total_recordings}"
    )

    print()

    results = []

    # =====================================================
    # PROCESS
    # =====================================================

    for index, (
        source_file,
        group,
    ) in enumerate(
        groups,
        start=1,
    ):

        group = (
            group
            .reset_index(
                drop=True
            )
        )

        # -------------------------------------------------
        # Participant
        # -------------------------------------------------

        participant = str(
            group[
                "participant"
            ].iloc[0]
        )

        # -------------------------------------------------
        # Exercise
        # -------------------------------------------------

        exercise = str(
            group[
                "exercise"
            ].iloc[0]
        ).lower().strip()

        # -------------------------------------------------
        # Intensity
        # -------------------------------------------------

        intensity = str(
            group[
                "intensity"
            ].iloc[0]
        ).lower().strip()

        # -------------------------------------------------
        # Duration
        # -------------------------------------------------

        duration_sec = (
            len(group)
            / FS
        )

        # -------------------------------------------------
        # Detect
        # -------------------------------------------------

        result = (
            count_repetitions_for_recording(
                group,
                FS,
            )
        )

        detected_reps = int(
            result[
                "detected_reps"
            ]
        )

        # -------------------------------------------------
        # Expected for evaluation ONLY
        # -------------------------------------------------

        expected_reps = (
            EXPECTED_REPS.get(
                intensity,
                np.nan,
            )
        )

        if pd.isna(
            expected_reps
        ):

            absolute_error = np.nan

        else:

            absolute_error = abs(
                detected_reps
                - int(
                    expected_reps
                )
            )

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        results.append(
            {

                "participant":
                    participant,

                "exercise":
                    exercise,

                "intensity":
                    intensity,

                "source_file":
                    source_file,

                "canonical_source":
                    canonical_source_name(
                        source_file
                    ),

                "duration_sec":
                    round(
                        duration_sec,
                        2,
                    ),

                "selected_axis":
                    result[
                        "selected_axis"
                    ],

                "estimated_period_sec":
                    result[
                        "estimated_period_sec"
                    ],

                "estimated_frequency_hz":
                    result[
                        "estimated_frequency_hz"
                    ],

                "period_score":
                    result[
                        "period_score"
                    ],

                "stability_score":
                    result[
                        "stability_score"
                    ],

                "axis_quality":
                    result[
                        "axis_quality"
                    ],

                "counting_method":
                    result[
                        "counting_method"
                    ],

                "candidate_counts":
                    result[
                        "candidate_counts"
                    ],

                "candidate_scores":
                    result[
                        "candidate_scores"
                    ],

                "detected_reps":
                    detected_reps,

                "expected_reps":
                    expected_reps,

                "absolute_error":
                    absolute_error,
            }
        )

        # -------------------------------------------------
        # PRINT
        # -------------------------------------------------

        print(
            f"[{index:03d}/"
            f"{total_recordings:03d}] "
            f"{participant:<2} | "
            f"{exercise:<8} | "
            f"{intensity:<9} | "
            f"Axis="
            f"{result['selected_axis']:<12} | "
            f"Period="
            f"{str(result['estimated_period_sec']):<5} | "
            f"Stable="
            f"{result['stability_score']:.2f} | "
            f"Final="
            f"{detected_reps:<2} | "
            f"Expected="
            f"{expected_reps}"
        )

    # =====================================================
    # DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            [
                "exercise",
                "participant",
                "source_file",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # SAVE MAIN RESULTS
    # =====================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # =====================================================
    # GLOBAL SUMMARY
    # =====================================================

    print()
    print("=" * 85)
    print(
        "REP COUNT SUMMARY"
    )
    print("=" * 85)

    valid_eval = (
        results_df[
            results_df[
                "expected_reps"
            ].notna()
        ]
        .copy()
    )

    if len(valid_eval) > 0:

        mae = float(
            valid_eval[
                "absolute_error"
            ].mean()
        )

        exact_match = float(
            valid_eval[
                "absolute_error"
            ]
            .eq(0)
            .mean()
            * 100
        )

        print(
            "Evaluation recordings : "
            f"{len(valid_eval)}"
        )

        print(
            "Mean absolute rep error: "
            f"{mae:.2f}"
        )

        print(
            "Exact rep-count match  : "
            f"{exact_match:.2f}%"
        )

    # =====================================================
    # EXERCISE-WISE
    # =====================================================

    print()
    print(
        "Exercise-wise results"
    )

    print(
        "-" * 85
    )

    for exercise in sorted(
        valid_eval[
            "exercise"
        ]
        .dropna()
        .unique()
    ):

        temp = valid_eval[
            valid_eval[
                "exercise"
            ]
            == exercise
        ]

        if len(temp) == 0:
            continue

        mae = float(
            temp[
                "absolute_error"
            ].mean()
        )

        exact = float(
            temp[
                "absolute_error"
            ]
            .eq(0)
            .mean()
            * 100
        )

        mean_detected = float(
            temp[
                "detected_reps"
            ].mean()
        )

        mean_expected = float(
            temp[
                "expected_reps"
            ].mean()
        )

        print(
            f"{exercise:<9} | "
            f"MAE={mae:.2f} | "
            f"Exact={exact:6.2f}% | "
            f"Detected={mean_detected:5.2f} | "
            f"Expected={mean_expected:5.2f}"
        )

    # =====================================================
    # INTENSITY-WISE
    # =====================================================

    print()
    print(
        "Intensity-wise results"
    )

    print(
        "-" * 85
    )

    for intensity in sorted(
        valid_eval[
            "intensity"
        ]
        .dropna()
        .unique()
    ):

        temp = valid_eval[
            valid_eval[
                "intensity"
            ]
            == intensity
        ]

        mae = float(
            temp[
                "absolute_error"
            ].mean()
        )

        exact = float(
            temp[
                "absolute_error"
            ]
            .eq(0)
            .mean()
            * 100
        )

        print(
            f"{intensity:<10} | "
            f"MAE={mae:.2f} | "
            f"Exact={exact:6.2f}%"
        )

    # =====================================================
    # PARTICIPANT-WISE
    # =====================================================

    print()
    print(
        "Participant-wise results"
    )

    print(
        "-" * 85
    )

    participant_summary = (
        create_participant_summary(
            results_df
        )
    )

    if len(
        participant_summary
    ) > 0:

        print(
            participant_summary.to_string(
                index=False
            )
        )

        participant_summary.to_csv(
            PARTICIPANT_SUMMARY_FILE,
            index=False,
        )

    # =====================================================
    # METHOD DISTRIBUTION
    # =====================================================

    print()
    print(
        "Counting-method distribution"
    )

    print(
        "-" * 85
    )

    method_counts = (
        results_df[
            "counting_method"
        ]
        .value_counts()
    )

    print(
        method_counts.to_string()
    )

    # =====================================================
    # AXIS DISTRIBUTION
    # =====================================================

    print()
    print(
        "Selected-axis distribution"
    )

    print(
        "-" * 85
    )

    axis_counts = (
        results_df[
            "selected_axis"
        ]
        .value_counts()
    )

    print(
        axis_counts.to_string()
    )

    # =====================================================
    # STABILITY DISTRIBUTION
    # =====================================================

    print()
    print(
        "Stability distribution"
    )

    print(
        "-" * 85
    )

    stability_values = (
        results_df[
            "stability_score"
        ]
        .dropna()
    )

    if len(
        stability_values
    ) > 0:

        print(
            "Mean stability : "
            f"{stability_values.mean():.3f}"
        )

        print(
            "Median stability: "
            f"{stability_values.median():.3f}"
        )

        print(
            "Min stability   : "
            f"{stability_values.min():.3f}"
        )

        print(
            "Max stability   : "
            f"{stability_values.max():.3f}"
        )

    # =====================================================
    # PERIOD SUMMARY
    # =====================================================

    print()
    print(
        "Estimated period by exercise"
    )

    print(
        "-" * 85
    )

    for exercise in sorted(
        results_df[
            "exercise"
        ]
        .dropna()
        .unique()
    ):

        temp = results_df[
            (
                results_df[
                    "exercise"
                ]
                == exercise
            )
            &
            (
                results_df[
                    "estimated_period_sec"
                ]
                .notna()
            )
        ]

        if len(temp) == 0:
            continue

        periods = temp[
            "estimated_period_sec"
        ].astype(float)

        print(
            f"{exercise:<9} | "
            f"mean={periods.mean():.2f}s | "
            f"median={periods.median():.2f}s | "
            f"min={periods.min():.2f}s | "
            f"max={periods.max():.2f}s"
        )

    # =====================================================
    # WORST 10
    # =====================================================

    print()
    print(
        "Worst 10 recordings"
    )

    print(
        "-" * 85
    )

    worst = (
        valid_eval
        .sort_values(
            "absolute_error",
            ascending=False,
        )
        [
            [
                "participant",
                "exercise",
                "intensity",
                "detected_reps",
                "expected_reps",
                "absolute_error",
                "selected_axis",
                "estimated_period_sec",
                "stability_score",
                "candidate_counts",
            ]
        ]
        .head(10)
    )

    if len(worst) > 0:

        print(
            worst.to_string(
                index=False
            )
        )

    # =====================================================
    # CONFIGURATION
    # =====================================================

    print()
    print(
        "V8 configuration"
    )

    print(
        "-" * 85
    )

    for exercise in BASE_CONFIG:

        print(
            exercise
        )

        for intensity in [
            "heavy",
            "medium",
        ]:

            config = (
                get_exercise_config(
                    exercise,
                    intensity,
                )
            )

            print(
                f"  {intensity:<7} | "
                f"period="
                f"{config['min_period_sec']:.2f}-"
                f"{config['max_period_sec']:.2f}s | "
                f"preferred="
                f"{config['preferred_period_sec']:.2f}s"
            )

    # =====================================================
    # SAVE MESSAGE
    # =====================================================

    print()
    print("=" * 85)
    print(
        "RESULTS SAVED"
    )
    print("=" * 85)

    print(
        OUTPUT_FILE
    )

    print(
        PARTICIPANT_SUMMARY_FILE
    )

    print()
    print(
        "REP COUNTER V8 COMPLETED ✅"
    )

    print()


# =========================================================
# 24. RUN
# =========================================================

if __name__ == "__main__":
    main()