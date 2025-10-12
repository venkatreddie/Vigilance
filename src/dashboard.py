import streamlit as st
import pandas as pd
import time
import sqlite3
import altair as alt

# -----------------------------
# DATABASE SETUP
# -----------------------------
DB_PATH = "vigilance_data.db"

def get_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT timestamp, event_type, confidence, status 
        FROM detection_logs 
        ORDER BY timestamp DESC 
        LIMIT 100
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error reading database: {e}")
        df = pd.DataFrame(columns=["timestamp", "event_type", "confidence", "status"])
    conn.close()
    return df


# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Driver Vigilance Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚘 Driver Vigilance Monitoring Dashboard")
st.caption("Real-time Monitoring and Analytics for Drowsiness, Yawning & Distraction Detection")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

refresh_rate = st.sidebar.slider("Auto Refresh (seconds)", 2, 30, 5)
show_raw = st.sidebar.checkbox("Show Raw Data", False)
show_chart = st.sidebar.checkbox("Show Charts", True)

# -----------------------------
# DASHBOARD DATA DISPLAY
# -----------------------------
placeholder = st.empty()

while True:
    with placeholder.container():
        df = get_data()

        col1, col2, col3 = st.columns(3)
        total_drowsy = df[df["event_type"] == "drowsy"].shape[0]
        total_yawn = df[df["event_type"] == "yawn"].shape[0]
        total_alert = df.shape[0]

        col1.metric("😴 Drowsiness Detected", total_drowsy)
        col2.metric("😮 Yawning Detected", total_yawn)
        col3.metric("📊 Total Alerts Logged", total_alert)

        # System Status
        st.markdown("### 🧠 System Status")
        if not df.empty:
            latest_status = df.iloc[0]["status"]
            status_color = "green" if latest_status == "Active" else "red"
            st.markdown(f"**Status:** <span style='color:{status_color}'>{latest_status}</span>", unsafe_allow_html=True)
        else:
            st.warning("No recent data found. Waiting for detection input...")

        # Charts
        if show_chart and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("count()", title="Event Count"),
                    color="event_type:N"
                )
                .properties(width="container", height=350, title="Event Frequency Over Time")
            )
            st.altair_chart(chart, use_container_width=True)

        # Raw Data
        if show_raw:
            st.markdown("### 📋 Recent Detection Logs")
            st.dataframe(df, use_container_width=True, height=300)

        st.markdown("---")
        st.markdown("© 2025 Driver Vigilance | Developed by Venkat")

    time.sleep(refresh_rate)
