import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import altair as alt

# -----------------------------
# 🧩 Page Setup
# -----------------------------
st.set_page_config(
    page_title="Driver Vigilance Monitoring Dashboard",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 Driver Vigilance Monitoring Dashboard")
st.markdown("### Real-time Monitoring and Analytics for Drowsiness, Yawning & Distraction Detection")

# -----------------------------
# 🗃️ Database Setup (Auto-Recovery)
# -----------------------------
DB_PATH = "detection_logs.db"

def connect_db():
    """Create or connect to the SQLite database safely."""
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                confidence REAL,
                status TEXT
            )
        """)
        conn.commit()
        conn.close()
    try:
        conn = sqlite3.connect(DB_PATH)
        # Test if valid by fetching 1 record
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return conn
    except Exception:
        # If corrupted, recreate
        os.remove(DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                confidence REAL,
                status TEXT
            )
        """)
        conn.commit()
        return conn

def get_data():
    """Fetch recent logs safely."""
    try:
        conn = connect_db()
        df = pd.read_sql_query(
            "SELECT timestamp, event_type, confidence, status FROM detection_logs ORDER BY timestamp DESC LIMIT 100",
            conn
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading database: {e}")
        return pd.DataFrame(columns=["timestamp", "event_type", "confidence", "status"])

# -----------------------------
# ⚙️ Control Panel
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    refresh_rate = st.slider("Auto Refresh (seconds)", 2, 30, 10)
    st.markdown("---")
    st.markdown("**Developed by Venkat 🚀**")

# -----------------------------
# 📊 Data Section
# -----------------------------
df = get_data()

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("😴 Drowsiness Detected", len(df[df["event_type"] == "Drowsiness"]))
with col2:
    st.metric("😮 Yawning Detected", len(df[df["event_type"] == "Yawning"]))
with col3:
    st.metric("📊 Total Alerts Logged", len(df))

# -----------------------------
# 🧠 System Status
# -----------------------------
st.subheader("🧠 System Status")

if not df.empty:
    latest_event = df.iloc[0]
    st.success(f"**Last Event:** {latest_event['event_type']} | **Status:** {latest_event['status']} | **Confidence:** {latest_event['confidence']:.2f}")
else:
    st.warning("No recent data found. Waiting for detection input...")

# -----------------------------
# 📈 Activity Graph
# -----------------------------
if not df.empty:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("confidence:Q", title="Confidence Level"),
            color=alt.Color("event_type:N", title="Event Type"),
            tooltip=["timestamp", "event_type", "confidence", "status"]
        )
        .properties(title="Recent Detection Confidence Trends", width="stretch", height=350)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No data available for graph visualization.")

# -----------------------------
# 🪵 Log Viewer
# -----------------------------
st.subheader("🪵 Recent Detection Logs")

if not df.empty:
    st.dataframe(df, use_container_width=True)
else:
    st.info("No detection logs found yet.")

# -----------------------------
# 🔁 Auto Refresh
# -----------------------------
st.markdown("### ⏳ Auto-refreshing in real-time...")
time.sleep(refresh_rate)
st.rerun()

# -----------------------------
# 🧾 Footer
# -----------------------------
st.markdown("---")
st.markdown("© 2025 **Driver Vigilance** | Developed by **Venkat**")
