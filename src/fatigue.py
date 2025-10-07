import cv2, mediapipe as mp, numpy as np, time, csv, os, math, winsound
from collections import deque
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------- CONFIG ----------
LOG_FILE = "log.csv"
DISTRACTION_DURATION = 10.0
YAWN_HOLD = 2.0
ALERT_REPEAT_INTERVAL = 2.0
ENTER_YAW_DEG, ENTER_PITCH_DEG = 20.0, 15.0
EXIT_YAW_DEG, EXIT_PITCH_DEG = 12.0, 8.0
CALIBRATE_SECONDS = 2.0
SMOOTH_WINDOW, GRAPH_LEN = 5, 60
BEEP_FREQ, BEEP_DUR_MS = 2000, 200

# create CSV header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Event", "Yaw", "Pitch", "Duration"])

# ---------- HELPERS ----------
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

def wrap_angle_deg(a): return (a + 180.0) % 360.0 - 180.0
def beep(): 
    try: winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
    except: pass

def draw_prob_graph(values, w=360, h=200):
    fig = Figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(values, color="red", linewidth=2)
    ax.set_ylim(0, 100); ax.set_xlim(0, GRAPH_LEN-1)
    ax.set_title("Fatigue Probability"); ax.grid(alpha=0.25)
    fig.tight_layout(); canvas = FigureCanvas(fig); canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# ---------- MOUTH ASPECT RATIO ----------
# landmark indices for upper & lower lips
OUTER_LIPS = [61, 291, 78, 308, 14, 13, 17, 0, 267, 402, 317, 82]
UPPER = [13, 14]
LOWER = [17, 0]
def mouth_aspect_ratio(lm, w, h):
    top = np.mean([(lm[i].x*w, lm[i].y*h) for i in UPPER], axis=0)
    bot = np.mean([(lm[i].x*w, lm[i].y*h) for i in LOWER], axis=0)
    left = np.array([lm[61].x*w, lm[61].y*h])
    right = np.array([lm[291].x*w, lm[291].y*h])
    dist_vert = np.linalg.norm(top - bot)
    dist_horiz = np.linalg.norm(left - right)
    return dist_vert / dist_horiz if dist_horiz > 0 else 0

MAR_THRESHOLD = 0.6  # adjust if needed

# ---------- FACE MESH SETUP ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
MODEL_POINTS = np.array([
    (0.0,0.0,0.0),(0.0,-330.0,-65.0),(-225.0,170.0,-135.0),
    (225.0,170.0,-135.0),(-150.0,-150.0,-125.0),(150.0,-150.0,-125.0)
],dtype=np.float64)
LMKS_IDX=[1,199,33,263,61,291]

# ---------- CALIBRATION ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise RuntimeError("Camera not found")

def calibrate_neutral(sec=CALIBRATE_SECONDS):
    print(f"[calib] Hold head straight for {sec}s...")
    y_,p_=[],[]
    t0=time.time()
    while time.time()-t0<sec:
        ret,frame=cap.read()
        if not ret: continue
        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark
            pts=np.array([(lm[i].x*w,lm[i].y*h) for i in LMKS_IDX],dtype=np.float64)
            cam=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float64)
            ok,rvec,tvec=cv2.solvePnP(MODEL_POINTS,pts,cam,np.zeros((4,1)))
            if ok:
                R,_=cv2.Rodrigues(rvec)
                _,p,y=rotationMatrixToEulerAngles(R)
                y_.append(y); p_.append(p)
        cv2.putText(frame,f"Calibrating {sec-(time.time()-t0):.1f}s",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        cv2.imshow("Calib",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):break
    cv2.destroyWindow("Calib")
    return (np.median(y_) if y_ else 0.0, np.median(p_) if p_ else 0.0)

base_yaw, base_pitch = calibrate_neutral()
print(f"[calib] yaw={base_yaw:.1f}, pitch={base_pitch:.1f}")

# ---------- STATE ----------
yaw_buf,pitch_buf=deque(maxlen=SMOOTH_WINDOW),deque(maxlen=SMOOTH_WINDOW)
prob_hist=deque([0]*GRAPH_LEN,maxlen=GRAPH_LEN)
dis_start=None;dis_alert=False;dis_logged=False
yawn_start=None;yawn_alert=False;yawn_logged=False
last_beep=0

# ---------- MAIN LOOP ----------
try:
    while True:
        ret,frame=cap.read()
        if not ret:break
        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark
            pts=np.array([(lm[i].x*w,lm[i].y*h) for i in LMKS_IDX],dtype=np.float64)
            cam=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float64)
            ok,rvec,tvec=cv2.solvePnP(MODEL_POINTS,pts,cam,np.zeros((4,1)))
            if ok:
                R,_=cv2.Rodrigues(rvec)
                roll,pitch,yaw=rotationMatrixToEulerAngles(R)
                yaw=wrap_angle_deg(yaw-base_yaw)
                pitch=pitch-base_pitch
                yaw_buf.append(yaw);pitch_buf.append(pitch)
                sy,sp=float(np.mean(yaw_buf)),float(np.mean(pitch_buf))
                cv2.putText(frame,f"YawΔ:{sy:+.1f}",(10,28),0,0.7,(0,255,255),2)
                cv2.putText(frame,f"PitchΔ:{sp:+.1f}",(10,56),0,0.7,(0,255,255),2)

                now=time.time()
                # ---- DISTRACTION ----
                distract=(abs(sy)>ENTER_YAW_DEG) or (sp>ENTER_PITCH_DEG)
                straight=(abs(sy)<=EXIT_YAW_DEG) and (abs(sp)<=EXIT_PITCH_DEG)
                if distract:
                    dis_start=dis_start or now
                    elapsed=now-dis_start
                else:
                    dis_start=None;elapsed=0
                    if dis_alert and straight:
                        with open(LOG_FILE,"a",newline="") as f:
                            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    "Distraction Ended",sy,sp,""])
                        dis_alert=False;dis_logged=False

                # ---- YAWN (MAR) ----
                mar=mouth_aspect_ratio(lm,w,h)
                cv2.putText(frame,f"MAR:{mar:.2f}",(10,84),0,0.7,(255,255,0),2)
                mouth_open=mar>MAR_THRESHOLD
                if mouth_open:
                    yawn_start=yawn_start or now
                    y_elapsed=now-yawn_start
                else:
                    y_elapsed=0;yawn_start=None
                    if yawn_alert:
                        with open(LOG_FILE,"a",newline="") as f:
                            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    "Yawn Ended",sy,sp,""])
                        yawn_alert=False;yawn_logged=False

                # ---- ALERT LOGIC ----
                if dis_start and (now-dis_start)>=DISTRACTION_DURATION:
                    if not dis_alert:
                        dis_alert=True
                        if not dis_logged:
                            with open(LOG_FILE,"a",newline="") as f:
                                csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                        "Distraction Started",sy,sp,""])
                            dis_logged=True
                        beep();last_beep=now
                    elif now-last_beep>=ALERT_REPEAT_INTERVAL:
                        beep();last_beep=now

                if yawn_start and (now-yawn_start)>=YAWN_HOLD:
                    if not yawn_alert:
                        yawn_alert=True
                        if not yawn_logged:
                            with open(LOG_FILE,"a",newline="") as f:
                                csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                        "Yawn Started",sy,sp,""])
                            yawn_logged=True
                        beep();last_beep=now
                    elif now-last_beep>=ALERT_REPEAT_INTERVAL:
                        beep();last_beep=now

                # ---- FATIGUE PROB ----
                p_disp=min(60,(elapsed/(DISTRACTION_DURATION*1.5))*60)
                p_yawn=min(40,(y_elapsed/(YAWN_HOLD*1.5))*40)
                prob_hist.append(int(min(100,p_disp+p_yawn)))

                # ---- WARNINGS ----
                if dis_alert or yawn_alert:
                    cv2.rectangle(frame,(60,40),(w-60,130),(0,0,255),-1)
                    txt="⚠ "
                    if dis_alert: txt+="DISTRACTION "
                    if yawn_alert: txt+="/ YAWN "
                    txt+="⚠"
                    cv2.putText(frame,txt,(80,100),0,0.9,(255,255,255),3)
        else:
            prob_hist.append(0)
            yaw_buf.clear();pitch_buf.clear()
            dis_alert=yawn_alert=False

        # ---- OVERLAYS ----
        poll=prob_hist[-1]
        bar_w=260
        cv2.rectangle(frame,(14,h-40),(14+bar_w,h-12),(180,180,180),2)
        cv2.rectangle(frame,(16,h-38),(16+int(poll/100*bar_w),h-14),(0,0,255),-1)
        cv2.putText(frame,f"Fatigue: {poll}%",(20+bar_w,h-18),0,0.7,(0,0,255),2)
        gimg=draw_prob_graph(prob_hist)
        gh,gw=gimg.shape[:2]; frame[10:10+gh,w-gw-10:w-10]=gimg
        cv2.putText(frame,"Press 'c' to recalibrate, 'q' to quit",
                    (12,h-6),0,0.5,(200,200,200),1)
        cv2.imshow("Fatigue Monitor - MAR",frame)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q') or k==27:break
        if k==ord('c'):
            base_yaw,base_pitch=calibrate_neutral()
            yaw_buf.clear();pitch_buf.clear()
            prob_hist.clear();prob_hist.extend([0]*GRAPH_LEN)
            print("[calib] recalibrated")

finally:
    cap.release();cv2.destroyAllWindows();face_mesh.close()
