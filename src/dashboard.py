import streamlit as st
import pandas as pd
import plotly.express as px
import os
import subprocess

st.set_page_config(page_title="Driver Vigilance Dashboard", layout="wide")

st.title("🚗 Driver Vigilance Analytics (Live Mode)")
st.markdown("Monitor driver distraction trends and performance in real-time.")

# CSV log file (auto-created by dash.py)
log_file = "detection_log.csv"

# ---- File Upload (Optional for testing) ----
uploaded_file = st.sidebar.file_uploader("Upload a log CSV (optional)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif os.path.exists(log_file):
    df = pd.read_csv(log_file)
else:
    df = pd.DataFrame()

# ---- Data Validation ----
required_cols = ["timestamp", "status"]
if df.empty:
    st.warning("No data found yet. Please start live detection.")
elif not all(col in df.columns for col in required_cols):
    st.error(f"CSV must contain columns: {required_cols}")
else:
    st.success("✅ Data loaded successfully!")

    # ---- Metrics ----
    total_frames = len(df)
    drowsy_count = df["status"].str.contains("Drowsy", case=False).sum()
    yawn_count = df["yawning"].sum() if "yawning" in df.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", total_frames)
    col2.metric("Drowsy Events", drowsy_count)
    col3.metric("Yawns Detected", yawn_count)

    st.divider()

    # ---- Plots ----
    if "timestamp" in df.columns and "status" in df.columns:
        fig1 = px.scatter(
            df, x="timestamp", y="status", color="status",
            title="Driver Vigilance Status Over Time"
        )
        st.plotly_chart(fig1, use_container_width=True)

    if "accident_probability" in df.columns:
        fig2 = px.histogram(df, x="accident_probability", nbins=10, title="Accident Probability Distribution")
        st.plotly_chart(fig2, use_container_width=True)

# ---- Control Panel ----
st.sidebar.markdown("## Controls")
if st.sidebar.button("▶ Start Live Detection"):
    try:
        subprocess.Popen(["python", "dash.py"])
        st.sidebar.success("Live detection started successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to start detection: {e}")

auto_refresh = st.sidebar.checkbox("Auto refresh every 5 seconds")
if auto_refresh:
    st.experimental_rerun()

st.markdown("---")
st.caption("Developed by Venkat • Driver Vigilance Monitoring System © 2025")
