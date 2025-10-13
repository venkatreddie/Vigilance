import streamlit as st
import sqlite3
import pandas as pd
import os

# -----------------------------
# Function to connect or recover database
# -----------------------------
def get_db_connection(db_path="vigilance.db"):
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("SELECT name FROM sqlite_master LIMIT 1;")
        return conn
    except sqlite3.DatabaseError:
        st.warning("⚠️ Database file corrupted. Recreating new database...")
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute('''CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                confidence REAL,
                status TEXT
            )''')
            conn.commit()
            st.success("✅ Database recovered successfully!")
            return conn
        except Exception as e:
            st.error(f"Database recovery failed: {e}")
            return None

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Driver Vigilance Dashboard",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 Driver Vigilance Monitoring Dashboard")
st.caption("Real-time Monitoring and Analytics for Drowsiness, Yawning & Distraction Detection")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")
refresh_interval = st.sidebar.slider("Auto Refresh Interval (seconds)", 2, 30, 5)

# This is the proper Streamlit method for live updates:
st_autorefresh = st.sidebar.checkbox("Enable Auto Refresh", value=True)
if st_autorefresh:
    st_autorefresh = st.experimental_rerun
    st_autorefresh = st.experimental_rerun
st_autorefresh_count = st.experimental_rerun if st_autorefresh else 0

# -----------------------------
# Connect or Recover DB
# -----------------------------
conn = get_db_connection()
if conn is None:
    st.stop()

# -----------------------------
# Load Detection Data
# -----------------------------
def load_data():
    try:
        df = pd.read_sql_query("""
            SELECT timestamp, event_type, confidence, status
            FROM detection_logs
            ORDER BY timestamp DESC
            LIMIT 100
        """, conn)
        return df
    except sqlite3.DatabaseError as e:
        if "file is not a database" in str(e).lower():
            st.warning("⚠️ Database corrupted during read. Recreating...")
            global conn
            conn = get_db_connection()
            return pd.DataFrame()
        else:
            st.error(f"Error reading database: {e}")
            return pd.DataFrame()

# -----------------------------
# Display Live Dashboard
# -----------------------------
df = load_data()

col1, col2, col3 = st.columns(3)
if not df.empty:
    drowsiness_count = len(df[df['event_type'] == 'drowsiness'])
    yawning_count = len(df[df['event_type'] == 'yawning'])
    total_alerts = len(df)

    with col1:
        st.metric("😴 Drowsiness Detected", drowsiness_count)
    with col2:
        st.metric("😮 Yawning Detected", yawning_count)
    with col3:
        st.metric("📊 Total Alerts Logged", total_alerts)

    st.subheader("🧠 System Status (Last 100 Records)")
    st.dataframe(df, width='stretch', height=300)

    # Trend chart
    st.subheader("📈 Detection Confidence Over Time")
    chart_df = df[['timestamp', 'confidence']]
    chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'], errors='coerce')
    chart_df = chart_df.dropna()
    if not chart_df.empty:
        st.line_chart(chart_df, x='timestamp', y='confidence')
    else:
        st.info("No valid timestamp data for chart.")
else:
    st.warning("No recent data found. Waiting for detection input...")

st.caption("© 2025 Driver Vigilance | Developed by Venkat")

# -----------------------------
# Auto-refresh logic
# -----------------------------
if st_autorefresh:
    import time
    time.sleep(refresh_interval)
    st.experimental_rerun()
