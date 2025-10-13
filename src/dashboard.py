import streamlit as st
import pandas as pd
import os
import time
import subprocess
import altair as alt
from datetime import datetime

# -----------------------------
# 🧩 Page Setup
# -----------------------------
st.set_page_config(
    page_title="Driver Vigilance Monitoring Dashboard",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 Driver Vigilance Monitoring Dashboard")
st.markdown("### Real-time Monitoring and Analytics for Distraction Detection")

# -----------------------------
# ⚙️ Control Panel (Sidebar)
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    refresh_rate = st.slider("Auto Refresh (seconds)", 2, 30, 5)
    st.markdown("---")
    st.subheader("🎥 Detection System")
    st.write("Click below to launch the live distraction detection camera system:")

    if st.button("▶️ Start Detection"):
        try:
            subprocess.Popen(["python", "distraction_detection.py"])
            st.success("✅ Detection system started successfully!")
        except Exception as e:
            st.error(f"Failed to start detection: {e}")

    st.markdown("---")
    st.markdown("**Developed by Venkat 🚀**")

# -----------------------------
# 📊 Data Section (CSV Logs)
# -----------------------------
LOG_FILE = "distraction_log.csv"
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
else:
    df = pd.DataFrame(columns=["Timestamp", "Yaw_deg", "Pitch_deg", "Event"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🟠 Distraction Started", len(df[df["Event"]=="Distraction Started"]))
with col2:
    st.metric("🟢 Distraction Ended", len(df[df["Event"]=="Distraction Ended"]))
with col3:
    st.metric("📊 Total Events Logged", len(df))

# -----------------------------
# 🧠 Latest Event
# -----------------------------
st.subheader("🧠 Last Detected Event")
if not df.empty:
    latest_event = df.iloc[-1]
    st.success(
        f"**Event:** {latest_event['Event']} | "
        f"**Yaw:** {latest_event['Yaw_deg']:.2f} | "
        f"**Pitch:** {latest_event['Pitch_deg']:.2f} | "
        f"**Timestamp:** {latest_event['Timestamp']}"
    )
else:
    st.info("No events logged yet.")

# -----------------------------
# 📈 Graph of Yaw & Pitch
# -----------------------------
if not df.empty:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Timestamp:T", title="Time"),
            y=alt.Y("Yaw_deg:Q", title="Yaw (deg)"),
            color=alt.Color("Event:N"),
            tooltip=["Timestamp", "Yaw_deg", "Pitch_deg", "Event"]
        )
        .properties(title="Head Yaw Trend")
    )
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# 🪵 Recent Logs
# -----------------------------
st.subheader("🪵 Recent Distraction Logs")
if not df.empty:
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
else:
    st.info("No logs found yet.")

# -----------------------------
# 🔁 Auto Refresh
# -----------------------------
st.markdown("### ⏳ Auto-refreshing in real-time...")
time.sleep(refresh_rate)
st.experimental_rerun()

# -----------------------------
# 🧾 Footer
# -----------------------------
st.markdown("---")
st.markdown("© 2025 **Driver Vigilance** | Developed by **Venkat**")
