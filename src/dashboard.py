import streamlit as st
import pandas as pd
import os
import subprocess
import altair as alt
import time

# -----------------------------
# 🧩 Page Setup
# -----------------------------
st.set_page_config(
    page_title="Driver Vigilance Monitoring Dashboard",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 Driver Vigilance Monitoring Dashboard")
st.markdown("### Real-time Monitoring and Analytics for Distraction, Drowsiness & Yawning")

# -----------------------------
# ⚙️ Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    refresh_rate = st.slider("Auto Refresh (seconds)", 2, 30, 5)
    st.markdown("---")
    st.subheader("🎥 Detection System")
    st.write("Click below to launch the live detection camera:")

    if st.button("▶️ Start Detection"):
        try:
            subprocess.Popen(["python", "detection_system.py"])
            st.success("✅ Detection system started successfully!")
        except Exception as e:
            st.error(f"Failed to start detection: {e}")

    st.markdown("---")
    st.markdown("**Developed by Venkat 🚀**")

# -----------------------------
# 🔁 Auto Refresh using timer
# -----------------------------
last_refresh = st.session_state.get("last_refresh", 0)
if time.time() - last_refresh > refresh_rate:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()

# -----------------------------
# 📊 Load CSV Logs
# -----------------------------
LOG_FILE = "distraction_log.csv"
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
else:
    df = pd.DataFrame(columns=["Timestamp", "Yaw_deg", "Pitch_deg", "EAR", "MAR", "Event"])

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🟠 Distraction Started", len(df[df["Event"]=="Distraction Started"]))
with col2:
    st.metric("🟢 Distraction Ended", len(df[df["Event"]=="Distraction Ended"]))
with col3:
    st.metric("😴 Drowsiness Detected", len(df[df["Event"]=="Drowsiness"]))
with col4:
    st.metric("😮 Yawning Detected", len(df[df["Event"]=="Yawning"]))

# -----------------------------
# Latest Event
# -----------------------------
st.subheader("🧠 Latest Event")
if not df.empty:
    latest_event = df.iloc[-1]
    st.success(
        f"**Event:** {latest_event['Event']} | "
        f"**Yaw:** {latest_event['Yaw_deg']} | "
        f"**Pitch:** {latest_event['Pitch_deg']} | "
        f"**EAR:** {latest_event['EAR']} | "
        f"**MAR:** {latest_event['MAR']} | "
        f"**Timestamp:** {latest_event['Timestamp']}"
    )
else:
    st.info("No events logged yet.")

# -----------------------------
# Graphs
# -----------------------------
if not df.empty:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Yaw & Pitch
    yaw_pitch_chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Timestamp:T",
            y=alt.Y("Yaw_deg:Q", title="Yaw (deg)"),
            color=alt.value("orange"),
            tooltip=["Timestamp", "Yaw_deg", "Pitch_deg", "Event"]
        )
        .properties(title="Head Yaw Trend", width="stretch", height=250)
    ) + (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Timestamp:T",
            y=alt.Y("Pitch_deg:Q", title="Pitch (deg)"),
            color=alt.value("cyan"),
            tooltip=["Timestamp", "Yaw_deg", "Pitch_deg", "Event"]
        )
    )
    st.altair_chart(yaw_pitch_chart, use_container_width=True)

    # EAR & MAR
    ear_mar_chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Timestamp:T",
            y="EAR:Q",
            color=alt.value("green"),
            tooltip=["Timestamp", "EAR", "MAR", "Event"]
        )
        .properties(title="Eye & Mouth Aspect Ratio Trend", width="stretch", height=250)
    ) + (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Timestamp:T",
            y="MAR:Q",
            color=alt.value("red"),
            tooltip=["Timestamp", "EAR", "MAR", "Event"]
        )
    )
    st.altair_chart(ear_mar_chart, use_container_width=True)

# -----------------------------
# Recent Logs
# -----------------------------
st.subheader("🪵 Recent Events")
if not df.empty:
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
else:
    st.info("No events logged yet.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("© 2025 **Driver Vigilance** | Developed by **Venkat**")
