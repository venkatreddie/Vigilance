import streamlit as st
import sqlite3
import pandas as pd
import time
import os

# -----------------------------
# Function to connect or recover database
# -----------------------------
def get_db_connection(db_path="vigilance.db"):
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        # Quick validity test
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
# Initialize Streamlit layout
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

auto_refresh = st.sidebar.checkbox("Auto Refresh (seconds)", value=True)
refresh_interval = st.sidebar.slider("Set Refresh Interval", 2, 30, 5)

# -----------------------------
# Connect or recover database
# -----------------------------
conn = get_db_connection()
if conn is None:
    st.stop()

# -----------------------------
# Function to fetch data
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
# Dashboard Metrics Section
# -----------------------------
placeholder = st.empty()

while True:
    with placeholder.container():
        df = load_data()

        if not df.empty:
            drowsiness_count = len(df[df['event_type'] == 'drowsiness'])
            yawning_count = len(df[df['event_type'] == 'yawning'])
            total_alerts = len(df)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("😴 Drowsiness Detected", drowsiness_count)
            with col2:
                st.metric("😮 Yawning Detected", yawning_count)
            with col3:
                st.metric("📊 Total Alerts Logged", total_alerts)

            st.subheader("🧠 System Status")
            st.dataframe(df, width='stretch', height=300)

        else:
            st.warning("No recent data found. Waiting for detection input...")

        st.caption("© 2025 Driver Vigilance | Developed by Venkat")

    if not auto_refresh:
        break
    time.sleep(refresh_interval)
