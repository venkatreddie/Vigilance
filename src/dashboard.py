# dashboard.py
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import winsound
import math
from collections import deque
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# -------- CONFIG --------
LOG_FILE = "distraction_log.csv"

# thresholds (degrees) relative to calibrated neutral pose
ENTER_YAW_DEG = 20.0       # enter distraction if abs(delta_yaw) > this
ENTER_PITCH_DEG = 15.0     # enter distraction if delta_pitch > this (looking down)

EXIT_YAW_DEG = 12.0        # consider straight if abs(delta_yaw) <= this
EXIT_PITCH_DEG = 8.0       # consider straight if delta_pitch <= this

DISTRACTION_DURATION = 10.0    # seconds continuous before alert
ALERT_REPEAT_INTERVAL = 2.0    # beep every N seconds while alert active

BEEP_FREQ = 2000      # Hz
BEEP_DUR_MS = 200     # ms

GRAPH_LEN = 50        # samples in plot
SMOOTH_WINDOW = 5     # moving average window for angles

CALIBRATE_SECONDS = 2.0  # seconds to auto-calibrate neutral pose at start

# -------- SETUP --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event", "Yaw_deg", "Pitch_deg", "Duration_s"])

# Model / landmark indices for solvePnP (same used previously)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye corner
    (225.0, 170.0, -135.0),   # right eye corner
    (-150.0, -150.0, -125.0), # left mouth
    (150.0, -150.0, -125.0)   # right mouth
], dtype=np.float64)

LMKS_IDX = [1, 199, 33, 263, 61, 291]  # indexes used from MediaPipe FaceMesh

# smoothing buffers
yaw_buf = deque(maxlen=SMOOTH_WINDOW)
pitch_buf = deque(maxlen=SMOOTH_WINDOW)

# logging / detection state
distraction_start_time = None
alert_active = False
logged_started = False
last_beep_time = 0.0

# baseline (neutral) values (set after calibration)
base_yaw = 0.0
base_pitch = 0.0
calibrated = False

# Probability history (for graph)
prob_history = deque(maxlen=GRAPH_LEN)

# helpers -------------------------------------------------------------------
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])   # roll
        y = math.atan2(-R[2,0], sy)      # pitch
        z = math.atan2(R[1,0], R[0,0])   # yaw
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])  # roll, pitch, yaw

def wrap_angle_deg(a):
    """Normalize angle to [-180, 180]"""
    a = (a + 180.0) % 360.0 - 180.0
    return a

def beep():
    try:
        winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
    except Exception:
        pass

def draw_probability_graph(values, width=360, height=200):
    """Render matplotlib plot to an OpenCV BGR image (in-memory)."""
    fig = Figure(figsize=(width/100.0, height/100.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(list(values), color="red", linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, GRAPH_LEN-1)
    ax.set_title("Distraction Probability")
    ax.set_ylabel("%")
    ax.set_xlabel("Samples")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    # RGBA -> BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img

# camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# calibration routine
def calibrate_neutral(seconds=CALIBRATE_SECONDS):
    """Capture neutral yaw/pitch for the given seconds and return median values."""
    print(f"[calib] Hold still for {seconds:.1f}s to set neutral pose...")
    samples_y = []
    samples_p = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            try:
                image_points = np.array([
                    (lm[LMKS_IDX[0]].x * w, lm[LMKS_IDX[0]].y * h),
                    (lm[LMKS_IDX[1]].x * w, lm[LMKS_IDX[1]].y * h),
                    (lm[LMKS_IDX[2]].x * w, lm[LMKS_IDX[2]].y * h),
                    (lm[LMKS_IDX[3]].x * w, lm[LMKS_IDX[3]].y * h),
                    (lm[LMKS_IDX[4]].x * w, lm[LMKS_IDX[4]].y * h),
                    (lm[LMKS_IDX[5]].x * w, lm[LMKS_IDX[5]].y * h)
                ], dtype=np.float64)

                focal_length = w
                center = (w/2.0, h/2.0)
                camera_matrix = np.array([[focal_length, 0, center[0]],
                                          [0, focal_length, center[1]],
                                          [0, 0, 1]], dtype=np.float64)
                dist_coeffs = np.zeros((4,1))
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    R_mat, _ = cv2.Rodrigues(rotation_vector)
                    _, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R_mat)
                    # normalize yaw to [-180,180]
                    yaw_deg = wrap_angle_deg(yaw_deg)
                    samples_y.append(yaw_deg)
                    samples_p.append(pitch_deg)
            except Exception:
                pass
        # show small message while calibrating
        cv2.putText(frame, f"Calibrating... {seconds - (time.time()-t0):.1f}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Distraction Dashboard - Calibrating", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # fallback if no samples
    if len(samples_y) == 0:
        return 0.0, 0.0
    return float(np.median(samples_y)), float(np.median(samples_p))

# initial calibration
base_yaw, base_pitch = calibrate_neutral(CALIBRATE_SECONDS)
calibrated = True
print(f"[calib] Done. base_yaw={base_yaw:.2f}, base_pitch={base_pitch:.2f}")
# ensure some initial graph values
for _ in range(GRAPH_LEN//2):
    prob_history.append(0)

# -------- main loop --------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        yaw_deg = 0.0
        pitch_deg = 0.0
        face_present = False

        if results.multi_face_landmarks:
            face_present = True
            lm = results.multi_face_landmarks[0].landmark
            try:
                image_points = np.array([
                    (lm[LMKS_IDX[0]].x * w, lm[LMKS_IDX[0]].y * h),
                    (lm[LMKS_IDX[1]].x * w, lm[LMKS_IDX[1]].y * h),
                    (lm[LMKS_IDX[2]].x * w, lm[LMKS_IDX[2]].y * h),
                    (lm[LMKS_IDX[3]].x * w, lm[LMKS_IDX[3]].y * h),
                    (lm[LMKS_IDX[4]].x * w, lm[LMKS_IDX[4]].y * h),
                    (lm[LMKS_IDX[5]].x * w, lm[LMKS_IDX[5]].y * h)
                ], dtype=np.float64)

                focal_length = w
                center = (w/2.0, h/2.0)
                camera_matrix = np.array([[focal_length, 0, center[0]],
                                          [0, focal_length, center[1]],
                                          [0, 0, 1]], dtype=np.float64)
                dist_coeffs = np.zeros((4,1))

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    R_mat, _ = cv2.Rodrigues(rotation_vector)
                    roll_deg, pitch_deg_raw, yaw_deg_raw = rotationMatrixToEulerAngles(R_mat)
                    # normalize yaw and compute delta relative to baseline
                    yaw_deg_raw = wrap_angle_deg(yaw_deg_raw)
                    # compute deltas relative to baseline
                    delta_yaw = wrap_angle_deg(yaw_deg_raw - base_yaw)
                    delta_pitch = pitch_deg_raw - base_pitch

                    # smoothing (short window)
                    yaw_buf.append(delta_yaw)
                    pitch_buf.append(delta_pitch)
                    smooth_yaw = float(np.mean(yaw_buf))
                    smooth_pitch = float(np.mean(pitch_buf))

                    # compute distraction/states using smoothed deltas
                    now = time.time()
                    is_distracted = (abs(smooth_yaw) > ENTER_YAW_DEG) or (smooth_pitch > ENTER_PITCH_DEG)
                    is_straight = (abs(smooth_yaw) <= EXIT_YAW_DEG) and (abs(smooth_pitch) <= EXIT_PITCH_DEG)

                    # manage timers & logging
                    if is_distracted:
                        if distraction_start_time is None:
                            distraction_start_time = now
                        elapsed = now - distraction_start_time
                        # compute simple probability from elapsed (0..DISTRACTION_DURATION*1.5 => 0..100)
                        prob = int(min(100.0, (elapsed / (DISTRACTION_DURATION * 1.5)) * 100.0))
                        prob_history.append(prob)
                        # activate alert only after hold duration
                        if elapsed >= DISTRACTION_DURATION:
                            if not alert_active:
                                alert_active = True
                                # immediate beep and log "started"
                                beep()
                                with open(LOG_FILE, "a", newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "Distraction Started", round(smooth_yaw,2), round(smooth_pitch,2), ""])
                                logged_started = True
                                last_beep_time = now
                            else:
                                # repeat beep every ALERT_REPEAT_INTERVAL
                                if (now - last_beep_time) >= ALERT_REPEAT_INTERVAL:
                                    beep()
                                    last_beep_time = now
                    else:
                        # not currently distracted: reset timer and probability
                        prob_history.append(0)
                        distraction_start_time = None
                        # if alert was active and now straight, end event and log end immediately
                        if alert_active and is_straight:
                            # log end and duration
                            with open(LOG_FILE, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 "Distraction Ended", round(smooth_yaw,2), round(smooth_pitch,2),
                                                 round((time.time() - last_beep_time), 1)])
                            # clear
                            alert_active = False
                            logged_started = False
                            last_beep_time = 0.0

                    # draw debug yaw/pitch (show deltas)
                    cv2.putText(frame, f"YawΔ: {smooth_yaw:+6.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(frame, f"PitchΔ: {smooth_pitch:+6.2f}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                else:
                    # solvePnP failed
                    prob_history.append(0)
            except Exception:
                prob_history.append(0)
        else:
            # no face
            prob_history.append(0)
            distraction_start_time = None
            alert_active = False
            logged_started = False
            yaw_buf.clear()
            pitch_buf.clear()
            cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Draw signboard if alert active
        if alert_active:
            cv2.rectangle(frame, (60, 40), (w-60, 130), (0, 0, 255), -1)
            cv2.putText(frame, "⚠ DISTRACTION DETECTED ⚠", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)

        # Draw probability bar (use last prob)
        poll = prob_history[-1] if len(prob_history) else 0
        bar_w = 220
        cv2.rectangle(frame, (12, h-36), (12+bar_w, h-12), (200,200,200), 2)
        fill_w = int((poll/100.0)*bar_w)
        cv2.rectangle(frame, (14, h-34), (14+fill_w, h-14), (0,0,255), -1)
        cv2.putText(frame, f"{poll}%", (14+bar_w+10, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Stats summary (scan log to produce quick stats if needed or keep counters)
        # For simplicity we won't rescan logs here; just show basic indicators.
        cv2.putText(frame, f"Alert: {'ON' if alert_active else 'OFF'}", (w-320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if not alert_active else (0,0,255), 2)

        # Draw graph overlay in top-right
        graph_img = draw_probability_graph(prob_history, width=360, height=200)
        gh, gw = graph_img.shape[:2]
        y0 = 10
        x0 = w - gw - 10
        # ensure overlay fits
        if x0 > 0 and y0 + gh < h:
            frame[y0:y0+gh, x0:x0+gw] = graph_img

        # small UI help
        cv2.putText(frame, "Press 'c' to recalibrate neutral pose, 'q' to quit", (10, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("Driver Vigilance - Dashboard", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('c'):  # manual recalibration
            base_yaw, base_pitch = calibrate_neutral(CALIBRATE_SECONDS)
            print(f"[calib] Recalibrated base_yaw={base_yaw:.2f}, base_pitch={base_pitch:.2f}")
            yaw_buf.clear()
            pitch_buf.clear()
            prob_history.clear()
            for _ in range(GRAPH_LEN//2):
                prob_history.append(0)

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
