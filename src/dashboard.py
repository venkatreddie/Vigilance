# dashboard.py  -- Streamlit dashboard that DOES NOT require mediapipe
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

st.set_page_config(page_title="Driver Vigilance Analytics", layout="wide")
st.title("🚗 Driver Vigilance Analytics (CSV-only)")

# Sidebar controls
st.sidebar.header("Controls")
upload = st.sidebar.file_uploader("Upload a log CSV (optional)", type=["csv"])
use_autorefresh = st.sidebar.checkbox("Auto refresh (requires streamlit-autorefresh)", value=False)
if use_autorefresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        # refresh every 5 seconds (5000 ms)
        st_autorefresh(interval=5_000, key="autorefresh")
    except Exception:
        st.sidebar.error("Install `streamlit-autorefresh` to enable auto refresh: pip install streamlit-autorefresh")

# search common filenames
CANDIDATES = [
    "distraction_log.csv",
    "data/distraction_log.csv",
    "log.csv",
    "fatigue_log.csv",
    "data/fatigue_log.csv",
    "distraction_log.csv"
]

found_path = None
for p in CANDIDATES:
    if os.path.exists(p):
        found_path = p
        break

if upload is not None:
    try:
        df = pd.read_csv(upload, encoding="utf-8", errors="replace")
        st.success("Loaded uploaded CSV")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
elif found_path:
    try:
        df = pd.read_csv(found_path, encoding="utf-8", errors="replace")
        st.success(f"Loaded log file: {found_path}")
    except Exception as e:
        st.error(f"Failed to read {found_path}: {e}")
        st.stop()
else:
    st.error(
        "No log file found. Place one of these files in the folder or upload one:\n\n"
        "- distraction_log.csv\n- data/distraction_log.csv\n- log.csv\n- fatigue_log.csv\n\n"
        "Or run your detection script to produce the CSV, then refresh this page."
    )
    st.stop()

# normalize column names (case-insensitive)
cols_lower = {c.lower(): c for c in df.columns}

def find_col(*options):
    for opt in options:
        if opt.lower() in cols_lower:
            return cols_lower[opt.lower()]
    return None

# possible column name variants
ts_col = find_col("Timestamp", "timestamp", "time", "Time")
yaw_col = find_col("Yaw", "Yaw_deg", "yaw", "yaw_deg")
pitch_col = find_col("Pitch", "Pitch_deg", "pitch", "pitch_deg")
mar_col = find_col("MAR", "mar")
event_col = find_col("Event", "event", "DistractionType", "Type")

# parse timestamp if exists
if ts_col:
    try:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    except Exception:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

# create standard columns for plotting if available
df_plot = df.copy()
df_plot["Timestamp"] = df_plot[ts_col] if ts_col in df_plot.columns else pd.NaT
if yaw_col:
    df_plot["Yaw"] = pd.to_numeric(df_plot[yaw_col], errors="coerce")
else:
    df_plot["Yaw"] = pd.NA
if pitch_col:
    df_plot["Pitch"] = pd.to_numeric(df_plot[pitch_col], errors="coerce")
else:
    df_plot["Pitch"] = pd.NA
if mar_col:
    df_plot["MAR"] = pd.to_numeric(df_plot[mar_col], errors="coerce")
else:
    df_plot["MAR"] = pd.NA
if event_col:
    df_plot["Event"] = df_plot[event_col]
else:
    # try to infer from other columns or set default
    df_plot["Event"] = df_plot.get("Event", df_plot.get("DistractionType", ""))

# Metrics
st.subheader("Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

total_events = len(df_plot)
avg_yaw = df_plot["Yaw"].dropna().mean() if df_plot["Yaw"].notna().any() else float("nan")
avg_pitch = df_plot["Pitch"].dropna().mean() if df_plot["Pitch"].notna().any() else float("nan")
avg_mar = df_plot["MAR"].dropna().mean() if df_plot["MAR"].notna().any() else float("nan")

col1.metric("Total events", total_events)
col2.metric("Avg Yaw", f"{avg_yaw:.2f}" if not pd.isna(avg_yaw) else "N/A")
col3.metric("Avg Pitch", f"{avg_pitch:.2f}" if not pd.isna(avg_pitch) else "N/A")
col4.metric("Avg MAR", f"{avg_mar:.2f}" if not pd.isna(avg_mar) else "N/A")

# Charts area
st.subheader("Yaw / Pitch / MAR over time")
fig, ax = plt.subplots(figsize=(12, 4))
has_plotted = False
if df_plot["Timestamp"].notna().any():
    if df_plot["Yaw"].notna().any():
        ax.plot(df_plot["Timestamp"], df_plot["Yaw"], label="Yaw", color="orange")
        has_plotted = True
    if df_plot["Pitch"].notna().any():
        ax.plot(df_plot["Timestamp"], df_plot["Pitch"], label="Pitch", color="blue")
        has_plotted = True
    if df_plot["MAR"].notna().any():
        ax.plot(df_plot["Timestamp"], df_plot["MAR"], label="MAR", color="green")
        has_plotted = True
    ax.set_xlabel("Time")
else:
    # fallback to index
    if df_plot["Yaw"].notna().any():
        ax.plot(df_plot.index, df_plot["Yaw"], label="Yaw", color="orange")
        has_plotted = True
    if df_plot["Pitch"].notna().any():
        ax.plot(df_plot.index, df_plot["Pitch"], label="Pitch", color="blue")
        has_plotted = True
    if df_plot["MAR"].notna().any():
        ax.plot(df_plot.index, df_plot["MAR"], label="MAR", color="green")
        has_plotted = True
    ax.set_xlabel("Record #")

if has_plotted:
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("No Yaw/Pitch/MAR numeric columns found to plot.")

# Frequency over time (per minute)
st.subheader("Events per minute")
if "Timestamp" in df_plot.columns and df_plot["Timestamp"].notna().any():
    df_counts = df_plot.set_index("Timestamp").resample("T").size()
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.bar(df_counts.index, df_counts.values, width=0.01, color="red")
    ax2.set_xlabel("Minute")
    ax2.set_ylabel("Event count")
    st.pyplot(fig2)
else:
    st.info("No timestamp column to compute per-minute counts.")

# Recent events table
st.subheader("Recent Events")
if "Timestamp" in df_plot.columns:
    df_recent = df_plot.sort_values("Timestamp", ascending=False).head(50)
else:
    df_recent = df_plot.tail(50)
st.dataframe(df_recent.reset_index(drop=True))

# Event breakdown (counts by Event column)
st.subheader("Event Types")
if "Event" in df_plot.columns and df_plot["Event"].notna().any():
    types = df_plot["Event"].value_counts()
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.pie(types.values, labels=types.index, autopct="%1.0f%%")
    ax3.axis("equal")
    st.pyplot(fig3)
else:
    st.info("No 'Event' column found in CSV (or it's empty).")

# Manual refresh button (Streamlit reruns script on widget interaction)
if st.button("Refresh"):
    st.experimental_rerun()

st.markdown("---")
st.caption("If you want the dashboard to reflect live camera detections, run the detection script to write the CSV (then press Refresh). "
           "To enable camera-based detection (which requires mediapipe), install mediapipe: `pip install mediapipe`.")
