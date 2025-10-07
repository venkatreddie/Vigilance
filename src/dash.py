import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# File path
LOG_FILE = "distraction_log.csv"

# Streamlit page settings
st.set_page_config(page_title="Driver Vigilance Analytics", layout="wide")

st.title("🚗 Driver Vigilance  Dashboard")
st.markdown("Monitor driver distraction trends and performance in real-time.")

# Check if log file exists
if not os.path.exists(LOG_FILE):
    st.error("⚠️ No log file found! Please run distraction_detection.py first.")
else:
    # Auto-refresh every 5 seconds
    st_autorefresh = st.sidebar.checkbox("🔄 Auto Refresh (Every 5s)", value=True)
    if st_autorefresh:
        time.sleep(5)
        st.experimental_rerun()

    # Load data
    df = pd.read_csv(LOG_FILE)

    if df.empty:
        st.warning("The log file is empty. No distraction data recorded yet.")
    else:
        # Convert Timestamp to datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        # Show summary metrics
        total_events = len(df)
        avg_yaw = df["Yaw"].mean()
        avg_pitch = df["Pitch"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Total Distraction Events", total_events)
        col2.metric("↔️ Avg Yaw", f"{avg_yaw:.2f}")
        col3.metric("↕️ Avg Pitch", f"{avg_pitch:.2f}")

        # Plot line chart (Yaw and Pitch over time)
        st.subheader("📈 Yaw and Pitch Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Timestamp"], df["Yaw"], label="Yaw", color="orange")
        ax.plot(df["Timestamp"], df["Pitch"], label="Pitch", color="blue")
        ax.set_xlabel("Time")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Distraction count chart
        st.subheader("📊 Distraction Frequency Over Time")
        df["Minute"] = df["Timestamp"].dt.floor("T")
        event_counts = df.groupby("Minute").size()

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(event_counts.index, event_counts.values, color="red", width=0.01)
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("Distraction Count")
        ax2.set_title("Distraction Events Frequency")
        st.pyplot(fig2)

        # Display last few distraction events
        st.subheader("🧾 Recent Events Log")
        st.dataframe(df.tail(10), use_container_width=True)
