from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="🏋️",
    layout="wide",
)


# =========================================================
# 2. PROJECT PATH
# =========================================================

# AI-Fitness-Tracker/
# ├── dashboard/
# │   └── app.py
# └── data/
#     └── processed/
#         ├── cleaned_accelerometer.csv
#         └── cleaned_gyroscope.csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

ACC_FILE = (
    PROCESSED_DIR / "cleaned_accelerometer.csv"
)

GYRO_FILE = (
    PROCESSED_DIR / "cleaned_gyroscope.csv"
)


# =========================================================
# 3. LOAD DATA
# =========================================================

@st.cache_data
def load_csv(file_path):

    return pd.read_csv(file_path)


# =========================================================
# 4. CHECK FILES
# =========================================================

if not ACC_FILE.exists() and not GYRO_FILE.exists():

    st.error("❌ Processed sensor files nahi mile.")

    st.write("Expected folder:")

    st.code(str(PROCESSED_DIR))

    st.stop()


# =========================================================
# 5. LOAD ACCELEROMETER
# =========================================================

acc_df = None

if ACC_FILE.exists():

    acc_df = load_csv(ACC_FILE)


# =========================================================
# 6. LOAD GYROSCOPE
# =========================================================

gyro_df = None

if GYRO_FILE.exists():

    gyro_df = load_csv(GYRO_FILE)


# =========================================================
# 7. SIDEBAR
# =========================================================

st.sidebar.title("🏋️ AI Fitness Tracker")

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset Status")

if acc_df is not None:

    st.sidebar.success(
        f"Accelerometer: {len(acc_df):,} rows"
    )

else:

    st.sidebar.warning(
        "Accelerometer file not found"
    )


if gyro_df is not None:

    st.sidebar.success(
        f"Gyroscope: {len(gyro_df):,} rows"
    )

else:

    st.sidebar.warning(
        "Gyroscope file not found"
    )


# =========================================================
# 8. MAIN HEADER
# =========================================================

st.title("🏋️ AI Fitness Tracker")

st.markdown(
    """
    ### Sensor-based Fitness Monitoring Dashboard

    Analyze accelerometer and gyroscope data collected
    during different workout exercises.
    """
)

st.markdown("---")


# =========================================================
# 9. TOP METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    if acc_df is not None:

        st.metric(
            "Accelerometer Records",
            f"{len(acc_df):,}"
        )

    else:

        st.metric(
            "Accelerometer Records",
            "N/A"
        )


with c2:

    if gyro_df is not None:

        st.metric(
            "Gyroscope Records",
            f"{len(gyro_df):,}"
        )

    else:

        st.metric(
            "Gyroscope Records",
            "N/A"
        )


with c3:

    if acc_df is not None and "exercise" in acc_df.columns:

        st.metric(
            "Exercises",
            acc_df["exercise"].nunique()
        )

    else:

        st.metric(
            "Exercises",
            "N/A"
        )


with c4:

    if acc_df is not None and "participant" in acc_df.columns:

        st.metric(
            "Participants",
            acc_df["participant"].nunique()
        )

    else:

        st.metric(
            "Participants",
            "N/A"
        )


# =========================================================
# 10. TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "📈 Accelerometer",
        "🔄 Gyroscope",
        "📋 Data",
    ]
)


# =========================================================
# TAB 1 - OVERVIEW
# =========================================================

with tab1:

    st.subheader("📊 Dataset Overview")


    # -----------------------------------------------------
    # Accelerometer overview
    # -----------------------------------------------------

    if acc_df is not None:

        st.markdown("### Accelerometer")

        st.write(
            f"Shape: **{acc_df.shape[0]:,} rows × "
            f"{acc_df.shape[1]} columns**"
        )

        st.dataframe(
            acc_df.head(10),
            use_container_width=True,
            hide_index=True
        )


        if "exercise" in acc_df.columns:

            st.subheader(
                "Exercise Distribution"
            )

            exercise_counts = (
                acc_df["exercise"]
                .value_counts()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                exercise_counts
            )


    # -----------------------------------------------------
    # Gyroscope overview
    # -----------------------------------------------------

    if gyro_df is not None:

        st.markdown("---")

        st.markdown("### Gyroscope")

        st.write(
            f"Shape: **{gyro_df.shape[0]:,} rows × "
            f"{gyro_df.shape[1]} columns**"
        )


# =========================================================
# TAB 2 - ACCELEROMETER
# =========================================================

with tab2:

    st.subheader(
        "📈 Accelerometer Sensor Analysis"
    )


    if acc_df is None:

        st.warning(
            "Accelerometer dataset available nahi hai."
        )

    else:

        st.write(
            f"Total records: **{len(acc_df):,}**"
        )


        acc_columns = [
            "x-axis (g)",
            "y-axis (g)",
            "z-axis (g)",
        ]


        existing_columns = [
            col
            for col in acc_columns
            if col in acc_df.columns
        ]


        if existing_columns:

            plot_columns = existing_columns.copy()


            if "elapsed (s)" in acc_df.columns:

                plot_df = acc_df[
                    ["elapsed (s)"] + plot_columns
                ].dropna(
                    subset=plot_columns,
                    how="all"
                )


                # Limit points for dashboard speed
                if len(plot_df) > 5000:

                    plot_df = plot_df.head(5000)


                if not plot_df.empty:

                    plot_df = plot_df.set_index(
                        "elapsed (s)"
                    )

                    st.line_chart(
                        plot_df
                    )

            else:

                plot_df = acc_df[
                    plot_columns
                ].dropna(
                    how="all"
                )


                if len(plot_df) > 5000:

                    plot_df = plot_df.head(5000)


                st.line_chart(
                    plot_df
                )


        else:

            st.warning(
                "Accelerometer axis columns nahi mile."
            )


# =========================================================
# TAB 3 - GYROSCOPE
# =========================================================

with tab3:

    st.subheader(
        "🔄 Gyroscope Sensor Analysis"
    )


    if gyro_df is None:

        st.warning(
            "Gyroscope dataset available nahi hai."
        )

    else:

        st.write(
            f"Total records: **{len(gyro_df):,}**"
        )


        gyro_columns = [
            "x-axis (deg/s)",
            "y-axis (deg/s)",
            "z-axis (deg/s)",
        ]


        existing_columns = [
            col
            for col in gyro_columns
            if col in gyro_df.columns
        ]


        if existing_columns:

            if "elapsed (s)" in gyro_df.columns:

                plot_df = gyro_df[
                    ["elapsed (s)"] + existing_columns
                ].dropna(
                    subset=existing_columns,
                    how="all"
                )


                if len(plot_df) > 5000:

                    plot_df = plot_df.head(5000)


                if not plot_df.empty:

                    plot_df = plot_df.set_index(
                        "elapsed (s)"
                    )

                    st.line_chart(
                        plot_df
                    )

            else:

                plot_df = gyro_df[
                    existing_columns
                ].dropna(
                    how="all"
                )


                if len(plot_df) > 5000:

                    plot_df = plot_df.head(5000)


                st.line_chart(
                    plot_df
                )


        else:

            st.warning(
                "Gyroscope axis columns nahi mile."
            )


# =========================================================
# TAB 4 - DATA
# =========================================================

with tab4:

    st.subheader(
        "📋 Dataset Preview"
    )


    dataset_choice = st.selectbox(
        "Select dataset",
        [
            "Accelerometer",
            "Gyroscope",
        ]
    )


    if dataset_choice == "Accelerometer":

        if acc_df is not None:

            st.write(
                f"Showing first 100 rows of "
                f"{len(acc_df):,}"
            )

            st.dataframe(
                acc_df.head(100),
                use_container_width=True,
                hide_index=True
            )


            st.subheader(
                "Missing Values"
            )

            st.dataframe(
                acc_df.isnull()
                .sum()
                .reset_index()
                .rename(
                    columns={
                        "index": "column",
                        0: "missing_values"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


    else:

        if gyro_df is not None:

            st.write(
                f"Showing first 100 rows of "
                f"{len(gyro_df):,}"
            )

            st.dataframe(
                gyro_df.head(100),
                use_container_width=True,
                hide_index=True
            )


            st.subheader(
                "Missing Values"
            )

            st.dataframe(
                gyro_df.isnull()
                .sum()
                .reset_index()
                .rename(
                    columns={
                        "index": "column",
                        0: "missing_values"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Fitness Tracker • Sensor Analytics Dashboard"
)