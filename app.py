from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from datetime import datetime
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

BACKEND_URL = "http://127.0.0.1:8000"
HEALTH_ENDPOINT = BACKEND_URL + "/health"
COUNT_ENDPOINT = BACKEND_URL + "/api/v1/sensor-data/count"
BACKEND_STREAM_FILE = (
    PROJECT_ROOT.parent
    / "AI-Fitness-Tracker-Phone-IMU"
    / "AI-Fitness-Tracker-Phone-IMU"
    / "backend"
    / "data"
    / "imu_stream.csv"
)

LIVE_SESSION_RAW_DIR = (
    PROJECT_ROOT / "data" / "raw"
)

PIPELINE_FILE = PROJECT_ROOT / "live_session_pipeline.py"


def run_live_session_pipeline(session_file):
    if not PIPELINE_FILE.exists():
        raise FileNotFoundError(
            f"Live session pipeline not found: {PIPELINE_FILE}"
        )

    with st.status(
        "Processing workout session...",
        expanded=False,
    ) as status:
        st.write(
            "Running preprocessing, exercise recognition, rep counting, form analysis, and history update."
        )

        subprocess.run(
            [
                sys.executable,
                str(PIPELINE_FILE),
                str(session_file),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            timeout=180,
        )

        status.update(
            label="Workout analysis completed.",
            state="complete",
            expanded=False,
        )

SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "workout_summary.csv"
REP_FILE = PROJECT_ROOT / "data" / "processed" / "phone_rep_consistency.csv"
REP_PREDICTION_FILE = PROJECT_ROOT / "data" / "processed" / "phone_rep_predictions.csv"
REP_PREDICTION_HISTORY_FILE = PROJECT_ROOT / "data" / "processed" / "phone_rep_prediction_history.csv"
SEGMENT_FILE = PROJECT_ROOT / "data" / "processed" / "phone_rep_segments.csv"
ACCELEROMETER_FILE = PROJECT_ROOT / "data" / "processed" / "phone_accelerometer_filtered.csv"
HISTORY_FILE = PROJECT_ROOT / "data" / "processed" / "workout_history.csv"
GYROSCOPE_FILE = PROJECT_ROOT / "data" / "processed" / "phone_gyroscope_filtered.csv"


st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
.stApp {
    background: #07101F;
    color: #F3F6FB;
}

header {
    visibility: hidden;
}

[data-testid="stSidebar"],
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: translateX(0) !important;
    width: 245px !important;
    min-width: 245px !important;
    max-width: 245px !important;
    background: #081426 !important;
    border-right: 1px solid #1D2A3D !important;
}

[data-testid="stSidebarContent"] {
    width: 245px !important;
    min-width: 245px !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 18px !important;
}

[data-testid="stSidebar"] .stButton {
    margin-bottom: 7px;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 40px;
    border-radius: 9px;
    border: 1px solid transparent;
    background: #0C1728;
    color: #9BAEC4;
    text-align: left;
    font-size: 11px;
    font-weight: 650;
}

[data-testid="stSidebar"] .stButton > button:hover {
    color: #FFFFFF;
    border-color: #347AF4;
    background: rgba(52, 122, 244, 0.16);
}

.block-container {
    max-width: 100%;
    padding: 16px 24px 28px 24px;
}

.sidebar-title {
    color: #F8FAFC;
    font-size: 20px;
    font-weight: 800;
}

.sidebar-subtitle {
    color: #7F93AA;
    font-size: 9px;
    margin-top: 4px;
    letter-spacing: 0.3px;
}

.sidebar-line {
    height: 1px;
    background: #1D2C42;
    margin: 15px 0;
}

.sidebar-about {
    margin-top: 110px;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #1E2D44;
    background: #0C1728;
}

.sidebar-about-title {
    color: #DDE7F2;
    font-size: 11px;
    font-weight: 700;
}

.sidebar-about-text {
    color: #71849A;
    font-size: 9px;
    line-height: 1.5;
    margin-top: 6px;
}

.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0 16px 0;
    border-bottom: 1px solid #1D2A3D;
    margin-bottom: 15px;
}

.brand-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-logo {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6D5DFB, #347AF4);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 17px;
}

.brand-title {
    color: #F8FAFC;
    font-size: 25px;
    font-weight: 820;
    line-height: 1.05;
}

.brand-subtitle {
    color: #89A0BA;
    font-size: 12px;
    margin-top: 3px;
}

.header-right {
    display: flex;
    align-items: center;
    gap: 12px;
}

.status-badge {
    color: #34D399;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.38);
    border-radius: 18px;
    padding: 7px 13px;
    font-size: 10px;
    font-weight: 650;
}

.header-time {
    color: #8E9FB3;
    font-size: 10px;
}

.section-title {
    color: #F4F7FB;
    font-size: 15px;
    font-weight: 760;
    margin: 5px 0 9px 0;
}

.card {
    background: #101B2C;
    border: 1px solid #1E2D44;
    border-radius: 13px;
    padding: 14px 15px;
    min-height: 104px;
    box-sizing: border-box;
}

.primary-card {
    background: linear-gradient(135deg, #1D3E91 0%, #111B2D 100%);
    border: 1px solid #347AF4;
    min-height: 122px;
}

.card-label {
    color: #91A5BD;
    font-size: 10px;
    font-weight: 550;
    margin-bottom: 8px;
}

.card-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: 820;
    line-height: 1.05;
}

.primary-value {
    color: #FFFFFF;
    font-size: 29px;
    font-weight: 850;
    margin-top: 5px;
}

.card-unit {
    color: #71849A;
    font-size: 10px;
    font-weight: 500;
}

.strength-tag {
    display: inline-block;
    color: #60A5FA;
    background: rgba(59, 130, 246, 0.13);
    border: 1px solid rgba(59, 130, 246, 0.40);
    border-radius: 7px;
    padding: 4px 8px;
    font-size: 8px;
    font-weight: 800;
    margin-top: 8px;
    letter-spacing: 0.3px;
}

.details-box {
    background: #0E1828;
    border: 1px solid #1E2D44;
    border-radius: 12px;
    padding: 10px 13px;
}

.detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid rgba(30, 45, 68, 0.48);
    font-size: 9px;
}

.detail-row:last-child {
    border-bottom: none;
}

.detail-label {
    color: #86A0B9;
}

.detail-value {
    color: #F2F7FC;
    font-weight: 650;
}

.page-box {
    background: #0E1828;
    border: 1px solid #1E2D44;
    border-radius: 13px;
    padding: 16px;
    margin-bottom: 14px;
}

.page-box-title {
    color: #F4F7FB;
    font-size: 14px;
    font-weight: 750;
    margin-bottom: 9px;
}

.page-box-text {
    color: #8EA2B8;
    font-size: 11px;
    line-height: 1.55;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1E2D44;
    border-radius: 10px;
    overflow: hidden;
}

.footer {
    text-align: center;
    color: #60748D;
    font-size: 8px;
    padding-top: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


def load_csv(path):
    if not path.exists():
        return None

    df = pd.read_csv(path)

    if df.empty:
        return None

    return df


def normalize_waveform(values, points=100):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 2:
        return None

    values = values - values.mean()

    std = values.std()

    if std <= 1e-12:
        return None

    values = values / std

    source_x = np.linspace(
        0.0,
        1.0,
        len(values),
    )

    target_x = np.linspace(
        0.0,
        1.0,
        points,
    )

    return np.interp(
        target_x,
        source_x,
        values,
    )





def save_live_session(start_count):
    if not BACKEND_STREAM_FILE.exists():
        raise FileNotFoundError(
            f"Backend stream file not found: {BACKEND_STREAM_FILE}"
        )

    df = pd.read_csv(
        BACKEND_STREAM_FILE
    )

    if df.empty:
        raise ValueError(
            "Backend stream contains no sensor samples."
        )

    start_index = max(
        0,
        int(start_count),
    )

    if start_index >= len(df):
        raise ValueError(
            "No new sensor samples were captured during this session."
        )

    session_df = df.iloc[
        start_index:
    ].copy()

    if session_df.empty:
        raise ValueError(
            "Current session contains no sensor samples."
        )

    LIVE_SESSION_RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        LIVE_SESSION_RAW_DIR
        / f"live_session_{session_id}.csv"
    )

    session_df.to_csv(
        output_file,
        index=False,
    )

    return output_file, len(session_df)


def load_live_sensor_data(max_rows=5000):
    if not BACKEND_STREAM_FILE.exists():
        return None

    try:
        df = pd.read_csv(
            BACKEND_STREAM_FILE
        )
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return None

    if df.empty:
        return None

    return df.tail(max_rows).copy()


def find_column(df, candidates):
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        candidate_lower = candidate.lower()

        if candidate_lower in normalized:
            return normalized[candidate_lower]

    for column in df.columns:
        name = str(column).strip().lower()

        for candidate in candidates:
            candidate_lower = candidate.lower()

            if candidate_lower in name:
                return column

    return None


def build_live_sensor_data(df):
    if df is None or df.empty:
        return None, None

    result = df.copy()

    timestamp_column = find_column(
        result,
        [
            "timestamp",
            "time",
            "elapsed",
            "epoch",
            "epoch_ms",
        ],
    )

    sensor_type_column = find_column(
        result,
        [
            "sensor_type",
            "sensor",
            "type",
        ],
    )

    accel_columns = {
        "X": find_column(
            result,
            [
                "accel_x",
                "accelerometer_x",
                "ax",
                "x_accel",
                "acceleration_x",
            ],
        ),
        "Y": find_column(
            result,
            [
                "accel_y",
                "accelerometer_y",
                "ay",
                "y_accel",
                "acceleration_y",
            ],
        ),
        "Z": find_column(
            result,
            [
                "accel_z",
                "accelerometer_z",
                "az",
                "z_accel",
                "acceleration_z",
            ],
        ),
    }

    gyro_columns = {
        "X": find_column(
            result,
            [
                "gyro_x",
                "gyroscope_x",
                "gx",
                "x_gyro",
                "angular_velocity_x",
            ],
        ),
        "Y": find_column(
            result,
            [
                "gyro_y",
                "gyroscope_y",
                "gy",
                "y_gyro",
                "angular_velocity_y",
            ],
        ),
        "Z": find_column(
            result,
            [
                "gyro_z",
                "gyroscope_z",
                "gz",
                "z_gyro",
                "angular_velocity_z",
            ],
        ),
    }

    accel_df = None
    gyro_df = None

    if sensor_type_column is not None:
        sensor_values = (
            result[sensor_type_column]
            .astype(str)
            .str.lower()
        )

        accel_mask = sensor_values.str.contains(
            "accel"
            + "|accelerometer",
            regex=True,
            na=False,
        )

        gyro_mask = sensor_values.str.contains(
            "gyro"
            + "|gyroscope",
            regex=True,
            na=False,
        )

        accel_df = result.loc[
            accel_mask
        ].copy()

        gyro_df = result.loc[
            gyro_mask
        ].copy()

    if accel_df is None or accel_df.empty:
        if all(
            value is not None
            for value in accel_columns.values()
        ):
            accel_df = result.copy()

    if gyro_df is None or gyro_df.empty:
        if all(
            value is not None
            for value in gyro_columns.values()
        ):
            gyro_df = result.copy()

    def prepare_frame(frame, columns):
        if frame is None or frame.empty:
            return None

        output = pd.DataFrame()

        for axis, column in columns.items():
            if column is None:
                return None

            output[axis] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        output = output.dropna(
            how="all"
        ).reset_index(
            drop=True
        )

        if output.empty:
            return None

        if timestamp_column in frame.columns:
            time_values = pd.to_numeric(
                frame[timestamp_column],
                errors="coerce",
            )

            if (
                time_values.notna().sum() >= 2
            ):
                time_values = time_values.reset_index(
                    drop=True
                )

                first_value = float(
                    time_values.iloc[0]
                )

                last_value = float(
                    time_values.iloc[-1]
                )

                if abs(last_value) > 1e11:
                    time_values = (
                        time_values - first_value
                    ) / 1000.0
                elif abs(last_value) > 1e9:
                    time_values = (
                        time_values - first_value
                    ) / 1000.0
                else:
                    time_values = (
                        time_values - first_value
                    )

                output["time"] = time_values

        if "time" not in output.columns:
            output["time"] = np.arange(
                len(output)
            )

        return output

    accel_prepared = prepare_frame(
        accel_df,
        accel_columns,
    )

    gyro_prepared = prepare_frame(
        gyro_df,
        gyro_columns,
    )

    return accel_prepared, gyro_prepared


def render_live_sensor_charts():
    live_df = load_live_sensor_data()

    accel_live, gyro_live = build_live_sensor_data(
        live_df
    )

    # Convert raw phone accelerometer from m/s^2 to g.
    if accel_live is not None:
        for axis in ["X", "Y", "Z"]:
            accel_live[axis] = (
                pd.to_numeric(
                    accel_live[axis],
                    errors="coerce",
                )
                / 9.80665
            )

        accel_live = accel_live.tail(
            300
        ).reset_index(
            drop=True
        )

    # Convert raw phone gyroscope from rad/s to deg/s.
    if gyro_live is not None:
        for axis in ["X", "Y", "Z"]:
            gyro_live[axis] = np.degrees(
                pd.to_numeric(
                    gyro_live[axis],
                    errors="coerce",
                )
            )

        gyro_live = gyro_live.tail(
            300
        ).reset_index(
            drop=True
        )

    if (
        accel_live is None
        and gyro_live is None
    ):
        st.info(
            "Waiting for sensor samples from the phone."
        )
        return

    st.markdown(
        '<div class="section-title">Live Sensor Graphs</div>',
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(
        2,
        gap="small",
    )

    with chart_col1:
        st.markdown(
            '<div class="page-card-title">Accelerometer</div>',
            unsafe_allow_html=True,
        )

        if accel_live is None:
            st.warning(
                "Accelerometer data is not available."
            )
        else:
            accel_fig = go.Figure()

            for axis in [
                "X",
                "Y",
                "Z",
            ]:
                accel_fig.add_trace(
                    go.Scatter(
                        x=accel_live["time"],
                        y=accel_live[axis],
                        mode="lines",
                        name=axis,
                    )
                )

            accel_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0E1828",
                height=380,
                margin=dict(
                    l=45,
                    r=15,
                    t=20,
                    b=45,
                ),
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Acceleration (g)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )

            st.plotly_chart(
                accel_fig,
                width="stretch",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                },
            )

    with chart_col2:
        st.markdown(
            '<div class="page-card-title">Gyroscope</div>',
            unsafe_allow_html=True,
        )

        if gyro_live is None:
            st.warning(
                "Gyroscope data is not available."
            )
        else:
            gyro_fig = go.Figure()

            for axis in [
                "X",
                "Y",
                "Z",
            ]:
                gyro_fig.add_trace(
                    go.Scatter(
                        x=gyro_live["time"],
                        y=gyro_live[axis],
                        mode="lines",
                        name=axis,
                    )
                )

            gyro_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0E1828",
                height=380,
                margin=dict(
                    l=45,
                    r=15,
                    t=20,
                    b=45,
                ),
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Angular Velocity (deg/s)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )

            st.plotly_chart(
                gyro_fig,
                width="stretch",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                },
            )


def backend_get_json(url, timeout=2):
    try:
        import json

        with urlopen(url, timeout=timeout) as response:
            return json.loads(
                response.read().decode("utf-8")
            )

    except (
        URLError,
        HTTPError,
        TimeoutError,
        OSError,
    ):
        return None


def get_backend_status():
    data = backend_get_json(
        HEALTH_ENDPOINT
    )

    if isinstance(data, dict):
        if data.get("status") == "ok":
            return True

        if data.get("service"):
            return True

    return False


def get_sensor_count():
    data = backend_get_json(
        COUNT_ENDPOINT
    )

    if data is None:
        return None

    if isinstance(data, (int, float)):
        return int(data)

    if isinstance(data, dict):
        possible_keys = [
            "count",
            "total",
            "total_count",
            "sensor_count",
            "samples",
            "rows",
        ]

        for key in possible_keys:
            value = data.get(key)

            if value is not None:
                try:
                    return int(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        nested = data.get("data")

        if isinstance(nested, dict):
            for key in possible_keys:
                value = nested.get(key)

                if value is not None:
                    try:
                        return int(value)
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

    return None


summary_df = load_csv(SUMMARY_FILE)
rep_df = load_csv(REP_FILE)
rep_prediction_df = load_csv(REP_PREDICTION_FILE)
rep_prediction_history_df = load_csv(REP_PREDICTION_HISTORY_FILE)
segment_df = load_csv(SEGMENT_FILE)
accel_df = load_csv(ACCELEROMETER_FILE)
history_df = load_csv(HISTORY_FILE)
gyro_df = load_csv(GYROSCOPE_FILE)


if summary_df is None:
    st.error("Workout summary file was not found.")
    st.stop()


summary = summary_df.iloc[0]

exercise = str(summary["exercise"]).upper()
reps = int(summary["detected_reps"])
active_duration = float(summary["active_duration_sec"])
cadence = float(summary["cadence_reps_per_min"])
mean_rep_duration = float(summary["mean_rep_duration_sec"])
rep_cv = float(summary["rep_duration_cv_percent"])
waveform_corr = float(summary["mean_waveform_correlation"])
pairwise_corr = float(summary["mean_pairwise_correlation"])
stability = float(summary["stability_score"])
axis_quality = float(summary["axis_quality"])
classifier_confidence = float(summary["classifier_confidence_percent"])
exercise_windows = float(summary["exercise_window_percentage"])
selected_axis = str(summary["selected_axis"])
period = float(summary["estimated_period_sec"])
frequency = float(summary["estimated_frequency_hz"])
amplitude_cv = float(summary["amplitude_range_cv_percent"])
weakest_rep_id = int(summary["weakest_rep_id"])
weakest_rep_corr = float(summary["weakest_rep_correlation"])


current_time = datetime.now().strftime(
    "%b %d, %Y - %I:%M %p"
)


if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"


with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">AI Fitness Tracker</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">PHONE IMU ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True,
    )

    dashboard_clicked = st.button(
        "D   Dashboard",
        key="sidebar_dashboard",
        width="stretch",
    )

    workout_clicked = st.button(
        "W   Workout Analysis",
        key="sidebar_workout",
        width="stretch",
    )

    rep_clicked = st.button(
        "R   Rep-by-Rep",
        key="sidebar_rep",
        width="stretch",
    )

    signal_clicked = st.button(
        "S   Signal Plots",
        key="sidebar_signal",
        width="stretch",
    )

    live_clicked = st.button(
        "L   Live Session",
        key="sidebar_live",
        width="stretch",
    )

    history_clicked = st.button(
        "H   History",
        key="sidebar_history",
        width="stretch",
    )

    export_clicked = st.button(
        "E   Export",
        key="sidebar_export",
        width="stretch",
    )

    if dashboard_clicked:
        st.session_state.active_page = "Dashboard"

    elif workout_clicked:
        st.session_state.active_page = "Workout Analysis"

    elif rep_clicked:
        st.session_state.active_page = "Rep-by-Rep"

    elif signal_clicked:
        st.session_state.active_page = "Signal Plots"

    elif live_clicked:
        st.session_state.active_page = "Live Session"

    elif history_clicked:
        st.session_state.active_page = "History"

    elif export_clicked:
        st.session_state.active_page = "Export"

    st.markdown(
        '<div class="sidebar-about"><div class="sidebar-about-title">About</div><div class="sidebar-about-text">Phone IMU based workout analytics for exercise recognition, repetition counting and movement consistency.</div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    f'<div class="top-header"><div class="brand-wrap"><div class="brand-logo">A</div><div><div class="brand-title">AI Fitness Tracker</div><div class="brand-subtitle">Phone IMU based workout analysis</div></div></div><div class="header-right"><div class="status-badge">Analysis Complete</div><div class="header-time">{current_time}</div></div></div>',
    unsafe_allow_html=True,
)


active_page = st.session_state.active_page


if active_page == "Dashboard":
    st.markdown(
        '<div class="section-title">Live Session Capture</div>',
        unsafe_allow_html=True,
    )

    if "live_monitoring" not in st.session_state:
        st.session_state.live_monitoring = False

    if "live_start_count" not in st.session_state:
        st.session_state.live_start_count = 0

    if "live_last_saved_file" not in st.session_state:
        st.session_state.live_last_saved_file = None

    if "live_last_saved_count" not in st.session_state:
        st.session_state.live_last_saved_count = 0

    current_live_count = 0

    try:
        if BACKEND_STREAM_FILE.exists():
            live_df = pd.read_csv(BACKEND_STREAM_FILE)
            current_live_count = len(live_df)
    except Exception:
        current_live_count = 0

    control_col1, control_col2, control_col3 = st.columns(
        [1.0, 1.0, 2.0],
        gap="small",
    )

    with control_col1:
        start_clicked = st.button(
            "Start Monitoring",
            key="start_live_monitoring",
            width="stretch",
            type="primary",
            disabled=st.session_state.live_monitoring,
        )

    with control_col2:
        stop_clicked = st.button(
            "Stop Monitoring",
            key="stop_live_monitoring",
            width="stretch",
            disabled=not st.session_state.live_monitoring,
        )

    with control_col3:
        if st.session_state.live_monitoring:
            st.info(
                f"Monitoring active | Backend samples: {current_live_count}"
            )
        else:
            st.caption(
                f"Backend samples available: {current_live_count}"
            )

    if start_clicked:
        try:
            if not BACKEND_STREAM_FILE.exists():
                st.error(
                    "Backend IMU stream file was not found."
                )
            else:
                live_df = pd.read_csv(
                    BACKEND_STREAM_FILE
                )

                st.session_state.live_start_count = len(
                    live_df
                )
                st.session_state.live_monitoring = True
                st.session_state.live_last_saved_file = None
                st.session_state.live_last_saved_count = 0

                st.success(
                    f"Monitoring started at sample {len(live_df)}."
                )

                st.rerun()

        except Exception as exc:
            st.error(
                f"Failed to start monitoring: {exc}"
            )

    if stop_clicked:
        try:
            saved_file, saved_count = save_live_session(
                st.session_state.live_start_count
            )

            run_live_session_pipeline(saved_file)

            st.session_state.live_monitoring = False
            st.session_state.live_last_saved_file = str(
                saved_file
            )
            st.session_state.live_last_saved_count = int(
                saved_count
            )

            st.success(
                f"Live session saved successfully: {saved_count} samples."
            )

            st.info(
                f"Saved file: {saved_file}"
            )

            st.rerun()

        except Exception as exc:
            st.session_state.live_monitoring = False
            st.error(
                f"Failed to save live session: {exc}"
            )

    if st.session_state.live_last_saved_file:
        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True,
        )

        st.write(
            "Last saved live session"
        )

        st.write(
            st.session_state.live_last_saved_file
        )

        st.write(
            f"Samples saved: {st.session_state.live_last_saved_count}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )


    st.markdown(
        '<div class="section-title">Workout Overview</div>',
        unsafe_allow_html=True,
    )

    overview_cols = st.columns(
        4,
        gap="small",
    )

    with overview_cols[0]:
        st.markdown(
            f'<div class="card primary-card"><div class="card-label">Exercise</div><div class="primary-value">{exercise}</div><div class="strength-tag">STRENGTH</div></div>',
            unsafe_allow_html=True,
        )

    with overview_cols[1]:
        st.markdown(
            f'<div class="card"><div class="card-label">Total Reps</div><div class="card-value">{reps} <span class="card-unit">repetitions</span></div></div>',
            unsafe_allow_html=True,
        )

    with overview_cols[2]:
        st.markdown(
            f'<div class="card"><div class="card-label">Active Duration</div><div class="card-value">{active_duration:.2f} <span class="card-unit">s</span></div></div>',
            unsafe_allow_html=True,
        )

    with overview_cols[3]:
        st.markdown(
            f'<div class="card"><div class="card-label">Cadence</div><div class="card-value">{cadence:.2f} <span class="card-unit">reps/min</span></div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    metrics = [
        ("Mean Rep Duration", f"{mean_rep_duration:.3f}", "s"),
        ("Rep Duration Variation", f"{rep_cv:.2f}", "%"),
        ("Waveform Correlation", f"{waveform_corr:.4f}", ""),
        ("Pairwise Correlation", f"{pairwise_corr:.4f}", ""),
        ("Stability Score", f"{stability:.4f}", ""),
        ("Axis Quality", f"{axis_quality:.4f}", ""),
        ("Classifier Confidence", f"{classifier_confidence:.2f}", "%"),
        ("Exercise Windows", f"{exercise_windows:.2f}", "%"),
    ]

    metric_cols = st.columns(
        8,
        gap="small",
    )

    for col, metric in zip(
        metric_cols,
        metrics,
    ):
        label, value, unit = metric

        with col:
            st.markdown(
                f'<div class="card"><div class="card-label">{label}</div><div class="card-value" style="font-size:20px;">{value} <span class="card-unit">{unit}</span></div></div>',
                unsafe_allow_html=True,
            )

    st.write("")

    chart_cols = st.columns(
        [1.65, 1.0],
        gap="small",
    )

    plot_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#98A9BE",
            size=10,
        ),
        margin=dict(
            l=45,
            r=12,
            t=24,
            b=38,
        ),
        height=315,
    )

    with chart_cols[0]:
        st.markdown(
            '<div class="section-title">IMU Signal with Detected Repetitions</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        if accel_df is not None:

            x = np.arange(
                len(accel_df)
            ) / 12.5

            traces = [
                ("ay_filtered", "Y axis", "#3B82F6", 2.0),
                ("ax_filtered", "X axis", "#8B5CF6", 1.4),
                ("az_filtered", "Z axis", "#10B981", 1.4),
            ]

            for column, name, line_color, line_width in traces:

                if column not in accel_df.columns:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=accel_df[column],
                        mode="lines",
                        name=name,
                        line=dict(
                            color=line_color,
                            width=line_width,
                        ),
                    )
                )

            if (
                segment_df is not None
                and "center_time_s" in segment_df.columns
            ):

                centers = pd.to_numeric(
                    segment_df["center_time_s"],
                    errors="coerce",
                ).dropna()

                for center in centers:

                    fig.add_vline(
                        x=float(center),
                        line_color="#EC4899",
                        line_dash="dash",
                        line_width=1,
                        opacity=0.85,
                    )

        fig.update_layout(
            **plot_layout,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        fig.update_xaxes(
            title="Time (seconds)",
            gridcolor="#1A2940",
            zeroline=False,
        )

        fig.update_yaxes(
            title="Acceleration (g)",
            gridcolor="#1A2940",
            zeroline=False,
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False,
            },
        )

    with chart_cols[1]:

        st.markdown(
            '<div class="section-title">Rep Waveform Consistency</div>',
            unsafe_allow_html=True,
        )

        fig_wave = go.Figure()
        waveforms = []

        if (
            segment_df is not None
            and accel_df is not None
            and selected_axis in accel_df.columns
        ):

            for _, row in segment_df.iterrows():

                start = int(
                    row["start_index"]
                )

                end = int(
                    row["end_index"]
                )

                signal = accel_df[
                    selected_axis
                ].iloc[
                    start:end + 1
                ].to_numpy()

                waveform = normalize_waveform(
                    signal
                )

                if waveform is None:
                    continue

                waveforms.append(
                    waveform
                )

                fig_wave.add_trace(
                    go.Scatter(
                        x=np.arange(100),
                        y=waveform,
                        mode="lines",
                        line=dict(
                            color="#557899",
                            width=1,
                        ),
                        opacity=0.28,
                        showlegend=False,
                    )
                )

        if waveforms:

            mean_waveform = np.mean(
                np.vstack(waveforms),
                axis=0,
            )

            fig_wave.add_trace(
                go.Scatter(
                    x=np.arange(100),
                    y=mean_waveform,
                    mode="lines",
                    name="Average Waveform",
                    line=dict(
                        color="#FFFFFF",
                        width=3,
                    ),
                )
            )

        fig_wave.update_layout(
            **plot_layout,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        fig_wave.update_xaxes(
            title="Normalized Time (%)",
            gridcolor="#1A2940",
        )

        fig_wave.update_yaxes(
            title="Normalized Amplitude",
            gridcolor="#1A2940",
            zeroline=False,
        )

        st.plotly_chart(
            fig_wave,
            width="stretch",
            config={
                "displayModeBar": False,
            },
        )

    st.write("")

    table_col, detail_col = st.columns(
        [1.45, 1.0],
        gap="small",
    )

    with table_col:

        st.markdown(
            '<div class="section-title">Rep-by-Rep Analysis</div>',
            unsafe_allow_html=True,
        )

        if rep_df is not None:

            table = rep_df.copy()

            if (
                rep_prediction_df is not None
                and "rep_id" in rep_prediction_df.columns
            ):
                prediction_columns = [
                    "rep_id",
                    "predicted_exercise",
                    "confidence_percent",
                ]

                prediction_columns = [
                    column
                    for column in prediction_columns
                    if column in rep_prediction_df.columns
                ]

                prediction_table = rep_prediction_df[
                    prediction_columns
                ].copy()

                table = table.merge(
                    prediction_table,
                    on="rep_id",
                    how="left",
                )

            columns = [
                "rep_id",
                "predicted_exercise",
                "confidence_percent",
                "duration_sec",
                "amplitude_range",
                "waveform_correlation",
            ]

            columns = [
                column
                for column in columns
                if column in table.columns
            ]

            table = table[
                columns
            ].copy()

            table = table.rename(
                columns={
                    "rep_id": "Rep ID",
                    "predicted_exercise": "Prediction",
                    "confidence_percent": "Confidence (%)",
                    "duration_sec": "Duration (s)",
                    "amplitude_range": "Amplitude Range",
                    "waveform_correlation": "Waveform Correlation",
                }
            )

            if "Confidence (%)" in table.columns:
                table["Confidence (%)"] = pd.to_numeric(
                    table["Confidence (%)"],
                    errors="coerce",
                ).round(2)

            st.dataframe(
                table.round(4),
                width="stretch",
                hide_index=True,
                height=330,
            )

        st.markdown(
            '<div class="section-title">Rep Prediction History</div>',
            unsafe_allow_html=True,
        )

        if (
            rep_prediction_history_df is not None
            and not rep_prediction_history_df.empty
        ):
            history_table = rep_prediction_history_df.copy()

            history_columns = [
                "session_id",
                "rep_id",
                "predicted_exercise",
                "confidence_percent",
                "duration_sec",
            ]

            history_columns = [
                column
                for column in history_columns
                if column in history_table.columns
            ]

            history_table = history_table[
                history_columns
            ].copy()

            history_table = history_table.sort_values(
                by=[
                    column
                    for column in ["session_id", "rep_id"]
                    if column in history_table.columns
                ],
                ascending=False,
            )

            history_table = history_table.rename(
                columns={
                    "session_id": "Session",
                    "rep_id": "Rep ID",
                    "predicted_exercise": "Prediction",
                    "confidence_percent": "Confidence (%)",
                    "duration_sec": "Duration (s)",
                }
            )

            if "Confidence (%)" in history_table.columns:
                history_table["Confidence (%)"] = pd.to_numeric(
                    history_table["Confidence (%)"],
                    errors="coerce",
                ).round(2)

            st.dataframe(
                history_table.round(4),
                width="stretch",
                hide_index=True,
                height=280,
            )
        else:
            st.info(
                "No per-repetition prediction history is available yet."
            )

    with detail_col:

        st.markdown(
            '<div class="section-title">Additional Metrics</div>',
            unsafe_allow_html=True,
        )

        details = [
            ("Exercise", exercise),
            ("Detected Reps", str(reps)),
            ("Active Duration", f"{active_duration:.2f} sec"),
            ("Cadence", f"{cadence:.2f} reps/min"),
            ("Selected Axis", selected_axis),
            ("Mean Rep Duration", f"{mean_rep_duration:.3f} sec"),
            ("Rep Duration Variation", f"{rep_cv:.2f}%"),
            ("Amplitude Range Variation", f"{amplitude_cv:.2f}%"),
            ("Mean Waveform Correlation", f"{waveform_corr:.4f}"),
            ("Mean Pairwise Correlation", f"{pairwise_corr:.4f}"),
            ("Stability Score", f"{stability:.4f}"),
            ("Axis Quality", f"{axis_quality:.4f}"),
            ("Classifier Confidence", f"{classifier_confidence:.2f}%"),
            ("Estimated Rep Period", f"{period:.3f} sec"),
            ("Estimated Frequency", f"{frequency:.4f} Hz"),
            ("Weakest Rep ID", str(weakest_rep_id)),
            ("Weakest Rep Correlation", f"{weakest_rep_corr:.4f}"),
        ]

        html = '<div class="details-box">'

        for label, value in details:

            html += (
                f'<div class="detail-row">'
                f'<span class="detail-label">{label}</span>'
                f'<span class="detail-value">{value}</span>'
                f'</div>'
            )

        html += "</div>"

        st.markdown(
            html,
            unsafe_allow_html=True,
        )


elif active_page == "Workout Analysis":

    st.markdown(
        '<div class="section-title">Workout Analysis</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(
        4,
        gap="small",
    )

    values = [
        ("Exercise", exercise),
        ("Detected Reps", str(reps)),
        ("Cadence", f"{cadence:.2f} reps/min"),
        ("Active Duration", f"{active_duration:.2f} sec"),
    ]

    for col, item in zip(
        cols,
        values,
    ):

        label, value = item

        with col:
            st.markdown(
                f'<div class="card"><div class="card-label">{label}</div><div class="card-value" style="font-size:21px;">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")

    st.markdown(
        f'<div class="page-box"><div class="page-box-title">Movement Consistency</div><div class="page-box-text">Mean waveform correlation: {waveform_corr:.4f}<br>Mean pairwise correlation: {pairwise_corr:.4f}<br>Rep duration variation: {rep_cv:.2f}%<br>Amplitude range variation: {amplitude_cv:.2f}%</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="page-box"><div class="page-box-title">Signal Quality</div><div class="page-box-text">Stability score: {stability:.4f}<br>Axis quality: {axis_quality:.4f}<br>Selected axis: {selected_axis}<br>Period score: {float(summary["period_score"]):.4f}</div></div>',
        unsafe_allow_html=True,
    )


elif active_page == "Rep-by-Rep":

    st.markdown(
        '<div class="section-title">Rep-by-Rep Analysis</div>',
        unsafe_allow_html=True,
    )

    if rep_df is None:

        st.warning(
            "Rep consistency data was not found."
        )

    else:

        table = rep_df.copy()

        table = table.rename(
            columns={
                "rep_id": "Rep ID",
                "duration_sec": "Duration (s)",
                "amplitude_range": "Amplitude Range",
                "waveform_correlation": "Waveform Correlation",
            }
        )

        st.dataframe(
            table.round(4),
            width="stretch",
            hide_index=True,
            height=500,
        )

        st.write("")

        rep_ids = rep_df[
            "rep_id"
        ].astype(int).tolist()

        selected_rep = st.selectbox(
            "Select repetition",
            rep_ids,
        )

        selected_row = rep_df[
            rep_df["rep_id"] == selected_rep
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Duration",
                f'{float(selected_row["duration_sec"]):.3f} s',
            )

        with c2:
            st.metric(
                "Amplitude Range",
                f'{float(selected_row["amplitude_range"]):.4f}',
            )

        with c3:
            st.metric(
                "Waveform Correlation",
                f'{float(selected_row["waveform_correlation"]):.4f}',
            )


elif active_page == "Signal Plots":

    st.markdown(
        '<div class="section-title">Signal Plots</div>',
        unsafe_allow_html=True,
    )

    if accel_df is None:

        st.warning(
            "Accelerometer data was not found."
        )

    else:

        st.markdown(
            '<div class="page-box"><div class="page-box-title">Accelerometer Analysis</div><div class="page-box-text">Inspect the filtered phone accelerometer signal and detected repetition locations.</div></div>',
            unsafe_allow_html=True,
        )

        control_col, info_col = st.columns(
            [1.0, 2.0],
            gap="small",
        )

        with control_col:

            selected_plot_axis = st.selectbox(
                "Signal axis",
                [
                    "ay_filtered",
                    "ax_filtered",
                    "az_filtered",
                ],
                format_func=lambda value: {
                    "ay_filtered": "Y axis",
                    "ax_filtered": "X axis",
                    "az_filtered": "Z axis",
                }[value],
            )

        signal_values = accel_df[
            selected_plot_axis
        ].astype(float)

        time_axis = np.arange(
            len(accel_df)
        ) / 12.5

        mean_value = float(
            signal_values.mean()
        )

        std_value = float(
            signal_values.std()
        )

        min_value = float(
            signal_values.min()
        )

        max_value = float(
            signal_values.max()
        )

        with info_col:

            info_cols = st.columns(
                4,
                gap="small",
            )

            with info_cols[0]:
                st.metric(
                    "Samples",
                    f"{len(signal_values):,}",
                )

            with info_cols[1]:
                st.metric(
                    "Mean",
                    f"{mean_value:.4f} g",
                )

            with info_cols[2]:
                st.metric(
                    "Standard Deviation",
                    f"{std_value:.4f} g",
                )

            with info_cols[3]:
                st.metric(
                    "Signal Range",
                    f"{min_value:.3f} to {max_value:.3f}",
                )

        st.write("")

        signal_fig = go.Figure()

        signal_fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_values,
                mode="lines",
                name="Filtered Signal",
                line=dict(
                    color="#3B82F6",
                    width=2,
                ),
            )
        )

        signal_fig.add_hline(
            y=mean_value,
            line_color="#64748B",
            line_dash="dot",
            line_width=1,
            annotation_text="Mean",
        )

        if (
            segment_df is not None
            and "center_time_s" in segment_df.columns
        ):

            centers = pd.to_numeric(
                segment_df["center_time_s"],
                errors="coerce",
            ).dropna()

            for index, center in enumerate(
                centers,
                start=1,
            ):

                signal_fig.add_vline(
                    x=float(center),
                    line_color="#EC4899",
                    line_dash="dash",
                    line_width=1,
                    opacity=0.75,
                )

                if 0 <= float(center) <= float(time_axis[-1]):

                    signal_fig.add_annotation(
                        x=float(center),
                        y=float(
                            signal_values.max()
                        ),
                        text=f"Rep {index}",
                        showarrow=False,
                        yshift=8,
                        font=dict(
                            color="#EC4899",
                            size=9,
                        ),
                    )

        signal_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0E1828",
            font=dict(
                color="#A7B6C8",
                size=10,
            ),
            height=560,
            margin=dict(
                l=55,
                r=25,
                t=30,
                b=55,
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        signal_fig.update_xaxes(
            title="Time (seconds)",
            gridcolor="#1A2940",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
        )

        signal_fig.update_yaxes(
            title="Acceleration (g)",
            gridcolor="#1A2940",
            zeroline=False,
        )

        st.plotly_chart(
            signal_fig,
            width="stretch",
            config={
                "displayModeBar": True,
                "displaylogo": False,
            },
        )

        st.markdown(
            '<div class="section-title">All Accelerometer Axes</div>',
            unsafe_allow_html=True,
        )

        all_axes_fig = go.Figure()

        axis_data = [
            (
                "ax_filtered",
                "X axis",
                "#8B5CF6",
            ),
            (
                "ay_filtered",
                "Y axis",
                "#3B82F6",
            ),
            (
                "az_filtered",
                "Z axis",
                "#10B981",
            ),
        ]

        for column, name, line_color in axis_data:

            if column not in accel_df.columns:
                continue

            all_axes_fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=accel_df[column],
                    mode="lines",
                    name=name,
                    line=dict(
                        color=line_color,
                        width=1.6,
                    ),
                )
            )

        all_axes_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0E1828",
            font=dict(
                color="#A7B6C8",
                size=10,
            ),
            height=420,
            margin=dict(
                l=55,
                r=25,
                t=30,
                b=55,
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        all_axes_fig.update_xaxes(
            title="Time (seconds)",
            gridcolor="#1A2940",
            zeroline=False,
        )

        all_axes_fig.update_yaxes(
            title="Acceleration (g)",
            gridcolor="#1A2940",
            zeroline=False,
        )

        st.plotly_chart(
            all_axes_fig,
            width="stretch",
            config={
                "displayModeBar": True,
                "displaylogo": False,
            },
        )

        if "phone_gyroscope_25hz.csv" in []:
            pass



    st.markdown(
        '<div class="section-title">Gyroscope Analysis</div>',
        unsafe_allow_html=True,
    )

    if gyro_df is None:

        st.warning(
            "Gyroscope data was not found."
        )

    else:

        gyro_controls, gyro_stats = st.columns(
            [1.0, 2.0],
            gap="small",
        )

        with gyro_controls:

            selected_gyro_axis = st.selectbox(
                "Gyroscope axis",
                [
                    "gx_filtered",
                    "gy_filtered",
                    "gz_filtered",
                ],
                format_func=lambda value: {
                    "gx_filtered": "GX axis",
                    "gy_filtered": "GY axis",
                    "gz_filtered": "GZ axis",
                }[value],
                key="gyro_axis_selector",
            )

        gyro_values = gyro_df[
            selected_gyro_axis
        ].astype(float)

        gyro_time = np.arange(
            len(gyro_df)
        ) / 25.0

        gyro_mean = float(
            gyro_values.mean()
        )

        gyro_std = float(
            gyro_values.std()
        )

        gyro_min = float(
            gyro_values.min()
        )

        gyro_max = float(
            gyro_values.max()
        )

        with gyro_stats:

            stat_cols = st.columns(
                4,
                gap="small",
            )

            with stat_cols[0]:
                st.metric(
                    "Samples",
                    f"{len(gyro_values):,}",
                )

            with stat_cols[1]:
                st.metric(
                    "Mean",
                    f"{gyro_mean:.3f} deg/s",
                )

            with stat_cols[2]:
                st.metric(
                    "Standard Deviation",
                    f"{gyro_std:.3f} deg/s",
                )

            with stat_cols[3]:
                st.metric(
                    "Signal Range",
                    f"{gyro_min:.1f} to {gyro_max:.1f}",
                )

        st.write("")

        gyro_fig = go.Figure()

        gyro_fig.add_trace(
            go.Scatter(
                x=gyro_time,
                y=gyro_values,
                mode="lines",
                name="Filtered Gyroscope",
                line=dict(
                    color="#F59E0B",
                    width=2,
                ),
            )
        )

        gyro_fig.add_hline(
            y=gyro_mean,
            line_color="#64748B",
            line_dash="dot",
            line_width=1,
            annotation_text="Mean",
        )

        if (
            segment_df is not None
            and "center_time_s" in segment_df.columns
        ):

            gyro_centers = pd.to_numeric(
                segment_df["center_time_s"],
                errors="coerce",
            ).dropna()

            for index, center in enumerate(
                gyro_centers,
                start=1,
            ):

                gyro_fig.add_vline(
                    x=float(center),
                    line_color="#EC4899",
                    line_dash="dash",
                    line_width=1,
                    opacity=0.75,
                )

                gyro_fig.add_annotation(
                    x=float(center),
                    y=float(gyro_values.max()),
                    text=f"Rep {index}",
                    showarrow=False,
                    yshift=8,
                    font=dict(
                        color="#EC4899",
                        size=9,
                    ),
                )

        gyro_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0E1828",
            font=dict(
                color="#A7B6C8",
                size=10,
            ),
            height=500,
            margin=dict(
                l=55,
                r=25,
                t=30,
                b=55,
            ),
            hovermode="x unified",
        )

        gyro_fig.update_xaxes(
            title="Time (seconds)",
            gridcolor="#1A2940",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
        )

        gyro_fig.update_yaxes(
            title="Angular Velocity (deg/s)",
            gridcolor="#1A2940",
            zeroline=False,
        )

        st.plotly_chart(
            gyro_fig,
            width="stretch",
            config={
                "displayModeBar": True,
                "displaylogo": False,
            },
        )

        st.markdown(
            '<div class="section-title">All Gyroscope Axes</div>',
            unsafe_allow_html=True,
        )

        all_gyro_fig = go.Figure()

        gyro_axes = [
            (
                "gx_filtered",
                "GX axis",
                "#F59E0B",
            ),
            (
                "gy_filtered",
                "GY axis",
                "#EF4444",
            ),
            (
                "gz_filtered",
                "GZ axis",
                "#A78BFA",
            ),
        ]

        for column, name, line_color in gyro_axes:

            if column not in gyro_df.columns:
                continue

            all_gyro_fig.add_trace(
                go.Scatter(
                    x=gyro_time,
                    y=gyro_df[column],
                    mode="lines",
                    name=name,
                    line=dict(
                        color=line_color,
                        width=1.6,
                    ),
                )
            )

        all_gyro_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0E1828",
            font=dict(
                color="#A7B6C8",
                size=10,
            ),
            height=420,
            margin=dict(
                l=55,
                r=25,
                t=30,
                b=55,
            ),
            hovermode="x unified",
        )

        all_gyro_fig.update_xaxes(
            title="Time (seconds)",
            gridcolor="#1A2940",
            zeroline=False,
        )

        all_gyro_fig.update_yaxes(
            title="Angular Velocity (deg/s)",
            gridcolor="#1A2940",
            zeroline=False,
        )

        st.plotly_chart(
            all_gyro_fig,
            width="stretch",
            config={
                "displayModeBar": True,
                "displaylogo": False,
            },
        )



elif active_page == "Live Session":

    @st.fragment(run_every=1)
    def live_session_page():

        st.markdown(
            '<div class="section-title">Live Session</div>',
            unsafe_allow_html=True,
        )

        backend_online = get_backend_status()
        sensor_count = get_sensor_count()

        if "live_monitoring" not in st.session_state:
            st.session_state.live_monitoring = False

        if "live_start_count" not in st.session_state:
            st.session_state.live_start_count = 0

        if "live_start_time" not in st.session_state:
            st.session_state.live_start_time = None

        if "live_last_saved_file" not in st.session_state:
            st.session_state.live_last_saved_file = None

        if "live_last_saved_count" not in st.session_state:
            st.session_state.live_last_saved_count = 0

        status_col, count_col, session_col = st.columns(
            3,
            gap="small",
        )

        with status_col:
            st.metric(
                "Backend Status",
                "Connected"
                if backend_online
                else "Offline",
            )

        with count_col:
            if sensor_count is None:
                st.metric(
                    "Sensor Samples",
                    "Unavailable",
                )
            else:
                st.metric(
                    "Sensor Samples",
                    f"{sensor_count:,}",
                )

        with session_col:
            if st.session_state.live_monitoring:
                current_session_count = 0

                if sensor_count is not None:
                    current_session_count = max(
                        0,
                        sensor_count
                        - st.session_state.live_start_count,
                    )

                st.metric(
                    "Current Session",
                    f"{current_session_count:,}",
                )
            else:
                st.metric(
                    "Current Session",
                    "Not Running",
                )

        st.markdown(
            '<div class="page-card">'
            '<div class="page-card-title">Phone IMU Monitor</div>'
            '<div class="page-card-text">'
            'Connect the Android phone and start the sensor stream. '
            'This page monitors data arriving through the FastAPI backend.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        control_col1, control_col2, control_col3 = st.columns(
            3,
            gap="small",
        )

        with control_col1:
            start_clicked = st.button(
                "Start Monitoring",
                width="stretch",
                disabled=st.session_state.live_monitoring,
                key="live_start_unified",
            )

        with control_col2:
            stop_clicked = st.button(
                "Stop Monitoring",
                width="stretch",
                disabled=not st.session_state.live_monitoring,
                key="live_stop_unified",
            )

        with control_col3:
            st.button(
                "Refresh Data",
                width="stretch",
                key="live_refresh_unified",
            )

        if start_clicked:
            current_count = get_sensor_count()

            st.session_state.live_monitoring = True

            st.session_state.live_start_count = (
                current_count
                if current_count is not None
                else 0
            )

            st.session_state.live_start_time = datetime.now()
            st.session_state.live_last_saved_file = None
            st.session_state.live_last_saved_count = 0

            st.rerun()

        if stop_clicked:
            try:
                saved_file, saved_count = save_live_session(
                    st.session_state.live_start_count
                )

                run_live_session_pipeline(saved_file)

                st.session_state.live_last_saved_file = str(
                    saved_file
                )

                st.session_state.live_last_saved_count = (
                    saved_count
                )

                st.session_state.live_monitoring = False
                st.session_state.live_start_time = None

                st.success(
                    f"Session captured: {saved_count} samples."
                )

                st.info(
                    f"Saved to: {saved_file}"
                )

            except (
                FileNotFoundError,
                ValueError,
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as error:
                st.error(
                    f"Session processing failed: {error}"
                )

            st.rerun()

        st.markdown(
            '<div class="section-title">Connection</div>',
            unsafe_allow_html=True,
        )

        connection_col1, connection_col2 = st.columns(
            2,
            gap="small",
        )

        with connection_col1:
            if backend_online:
                st.success(
                    "FastAPI backend is connected."
                )
            else:
                st.error(
                    "FastAPI backend is not reachable."
                )

        with connection_col2:
            if st.session_state.live_monitoring:
                st.success(
                    "Live monitoring is active."
                )
            else:
                st.info(
                    "Live monitoring is idle."
                )

        if (
            st.session_state.live_monitoring
            and st.session_state.live_start_time is not None
        ):
            elapsed = (
                datetime.now()
                - st.session_state.live_start_time
            ).total_seconds()

            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            st.markdown(
                '<div class="page-card">'
                '<div class="page-card-title">Session Timer</div>'
                f'<div class="primary-value">'
                f'{minutes:02d}:{seconds:02d}'
                f'</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-title">Live Sensor Graphs</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.live_monitoring:
            render_live_sensor_charts()
        else:
            st.info(
                "Start monitoring to display live sensor graphs."
            )

        st.markdown(
            '<div class="section-title">Sensor Status</div>',
            unsafe_allow_html=True,
        )

        sensor_status_col1, sensor_status_col2 = st.columns(
            2,
            gap="small",
        )

        with sensor_status_col1:
            st.markdown(
                '<div class="page-card">'
                '<div class="page-card-title">Accelerometer</div>'
                '<div class="page-card-text">'
                'Phone accelerometer data is sent to the backend.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with sensor_status_col2:
            st.markdown(
                '<div class="page-card">'
                '<div class="page-card-title">Gyroscope</div>'
                '<div class="page-card-text">'
                'Phone gyroscope data is sent to the backend.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    live_session_page()

elif active_page == "History":

    st.markdown(
        '<div class="section-title">Workout History</div>',
        unsafe_allow_html=True,
    )

    if history_df is None:

        st.warning(
            "No workout history is available."
        )

    else:

        history_table = history_df.copy()

        display_columns = [
            "session_id",
            "date",
            "time",
            "exercise",
            "detected_reps",
            "active_duration_sec",
            "cadence_reps_per_min",
            "mean_rep_duration_sec",
            "rep_duration_cv_percent",
            "mean_waveform_correlation",
            "stability_score",
            "classifier_confidence_percent",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in history_table.columns
        ]

        history_table = history_table[
            display_columns
        ].copy()

        history_table = history_table.rename(
            columns={
                "session_id": "Session ID",
                "date": "Date",
                "time": "Time",
                "exercise": "Exercise",
                "detected_reps": "Reps",
                "active_duration_sec": "Active Duration (s)",
                "cadence_reps_per_min": "Cadence (reps/min)",
                "mean_rep_duration_sec": "Mean Rep Duration (s)",
                "rep_duration_cv_percent": "Rep Duration Variation (%)",
                "mean_waveform_correlation": "Waveform Correlation",
                "stability_score": "Stability Score",
                "classifier_confidence_percent": "Classifier Confidence (%)",
            }
        )

        numeric_columns = [
            "Active Duration (s)",
            "Cadence (reps/min)",
            "Mean Rep Duration (s)",
            "Rep Duration Variation (%)",
            "Waveform Correlation",
            "Stability Score",
            "Classifier Confidence (%)",
        ]

        for column in numeric_columns:
            if column in history_table.columns:
                history_table[column] = history_table[
                    column
                ].round(3)

        st.dataframe(
            history_table,
            width="stretch",
            hide_index=True,
            height=420,
        )

        st.write("")

        st.markdown(
            '<div class="section-title">Selected Workout</div>',
            unsafe_allow_html=True,
        )

        session_ids = history_df[
            "session_id"
        ].astype(str).tolist()

        selected_session = st.selectbox(
            "Select session",
            session_ids,
            key="history_session_selector",
        )

        selected_history = history_df[
            history_df["session_id"].astype(str)
            == selected_session
        ].iloc[0]

        selected_exercise = str(
            selected_history["exercise"]
        ).upper()

        selected_reps = int(
            selected_history["detected_reps"]
        )

        selected_duration = float(
            selected_history["active_duration_sec"]
        )

        selected_cadence = float(
            selected_history["cadence_reps_per_min"]
        )

        selected_corr = float(
            selected_history["mean_waveform_correlation"]
        )

        selected_stability = float(
            selected_history["stability_score"]
        )

        selected_confidence = float(
            selected_history["classifier_confidence_percent"]
        )

        selected_cols = st.columns(
            6,
            gap="small",
        )

        selected_metrics = [
            (
                "Exercise",
                selected_exercise,
            ),
            (
                "Reps",
                str(selected_reps),
            ),
            (
                "Duration",
                f"{selected_duration:.2f} s",
            ),
            (
                "Cadence",
                f"{selected_cadence:.2f}",
            ),
            (
                "Correlation",
                f"{selected_corr:.4f}",
            ),
            (
                "Stability",
                f"{selected_stability:.4f}",
            ),
        ]

        for col, item in zip(
            selected_cols,
            selected_metrics,
        ):

            label, value = item

            with col:

                st.markdown(
                    f'<div class="card"><div class="card-label">{label}</div><div class="card-value" style="font-size:20px;">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")

        st.markdown(
            f'<div class="page-box"><div class="page-box-title">Session Details</div><div class="page-box-text">Session ID: {selected_session}<br>Date: {selected_history["date"]}<br>Time: {selected_history["time"]}<br>Exercise windows: {float(selected_history["classifier_confidence_percent"]):.2f}% classifier confidence<br>Classifier confidence: {selected_confidence:.2f}%</div></div>',
            unsafe_allow_html=True,
        )



elif active_page == "Export":

    st.markdown(
        '<div class="section-title">Export</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-box"><div class="page-box-title">Workout Summary</div><div class="page-box-text">Export the current workout summary and repetition-level analysis.</div></div>',
        unsafe_allow_html=True,
    )

    summary_bytes = summary_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Workout Summary",
        data=summary_bytes,
        file_name="workout_summary.csv",
        mime="text/csv",
        width="stretch",
    )

    if rep_df is not None:

        rep_bytes = rep_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Rep Consistency",
            data=rep_bytes,
            file_name="phone_rep_consistency.csv",
            mime="text/csv",
            width="stretch",
        )


st.markdown(
    '<div class="footer">AI Fitness Tracker | Phone IMU Workout Analytics</div>',
    unsafe_allow_html=True,
)
