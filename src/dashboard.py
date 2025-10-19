import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os
import threading
from datetime import datetime

# ============== Sound Alert Setup ==============
try:
    import winsound
except:
    winsound = None

# Global alarm control
alarm_active = False
alarm_thread = None

def play_continuous_beep():
    """Continuously beep until alarm_active is False."""
    global alarm_active
    while alarm_active:
        if winsound:
            try:
                winsound.Beep(1800, 400)  # frequency, duration
            except:
                pass
        time.sleep(0.1)  # short delay between beeps

def start_alarm():
    """Start the continuous beep thread."""
    global alarm_active, alarm_thread
    if not alarm_active:
        alarm_active = True
        alarm_thread = threading.Thread(target=play_continuous_beep, daemon=True)
        alarm_thread.start()

def stop_alarm():
    """Stop the continuous beep."""
    global alarm_active
    alarm_active = False

# ============== Detection Config ==============
LOG_FILE = "detection_log.csv"
ALERT_DELAY = 5.0
EAR_THRESHOLD = 0.25
YAWN_RATIO_THRESHOLD = 0.45
DISTRACTION_YAW = 20.0
DISTRACTION_PITCH = 15.0

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
MOUTH_TOP_INNER = [13,14]
MOUTH_BOTTOM_INNER = [17,18]
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

MODEL_POINTS = np.array([
    (0.0,0.0,0.0),
    (0.0,-330.0,-65.0),
    (-225.0,170.0,-135.0),
    (225.0,170.0,-135.0),
    (-150.0,-150.0,-125.0),
    (150.0,-150.0,-125.0)
],dtype=np.float64)
LMKS_IDX = [1,199,33,263,61,291]

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,"w",newline="") as f:
        csv.writer(f).writerow(["Timestamp","Event","Yaw_deg","Pitch_deg","EAR","MouthRatio"])

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2+R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1],R[2,2])
        y = math.atan2(-R[2,0],sy)
        z = math.atan2(R[1,0],R[0,0])
    else:
        x = math.atan2(-R[1,2],R[1,1])
        y = math.atan2(-R[2,0],sy)
        z = 0
    return np.degrees([x,y,z])

def eye_aspect_ratio(lm, li, ri):
    def ear(p):
        A = np.linalg.norm(p[1]-p[5])
        B = np.linalg.norm(p[2]-p[4])
        C = np.linalg.norm(p[0]-p[3])
        return (A+B)/(2.0*C) if C!=0 else 0.0
    left = np.array([lm[i] for i in li])
    right = np.array([lm[i] for i in ri])
    return (ear(left)+ear(right))/2.0

def mouth_open_ratio(lm):
    top = [lm[i] for i in MOUTH_TOP_INNER if i<len(lm)]
    bottom = [lm[i] for i in MOUTH_BOTTOM_INNER if i<len(lm)]
    if not top or not bottom:
        return 0.0
    verts = [math.hypot(t[0]-b[0],t[1]-b[1]) for t,b in zip(top,bottom)]
    vertical = np.mean(verts)
    left_corner = lm[MOUTH_LEFT_CORNER]; right_corner = lm[MOUTH_RIGHT_CORNER]
    horizontal = math.hypot(left_corner[0]-right_corner[0],left_corner[1]-right_corner[1]) or 1.0
    return vertical/horizontal

def log_event(event,yaw,pitch,ear,mr):
    ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([ts,event,round(yaw,2),round(pitch,2),round(ear,3),round(mr,3)])
    return {"Timestamp":ts,"Event":event,"Yaw":round(yaw,2),"Pitch":round(pitch,2),"EAR":round(ear,3),"MouthRatio":round(mr,3)}

# ============== Streamlit UI ==============
st.set_page_config(page_title="Driver Vigilance Dashboard",layout="wide")
st.title("🚘 Driver Vigilance Detection — Live Monitoring with Continuous Alarm")

start=st.sidebar.button("▶ Start Detection")
stop =st.sidebar.button("⏹ Stop Detection")

if "running" not in st.session_state:
    st.session_state.running=False
if start:
    st.session_state.running=True
if stop:
    st.session_state.running=False
    stop_alarm()

frame_slot=st.empty()
status_slot=st.empty()
mcol1,mcol2,mcol3,mcol4=st.columns(4)
m1=mcol1.empty();m2=mcol2.empty();m3=mcol3.empty();m4=mcol4.empty()

if "start_times" not in st.session_state:
    st.session_state.start_times={"Drowsiness":None,"Yawning":None,"Distraction":None}
if "alerted" not in st.session_state:
    st.session_state.alerted={"Drowsiness":False,"Yawning":False,"Distraction":False}

if st.session_state.running:
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera.")
        st.session_state.running=False
    else:
        with mp_face_mesh.FaceMesh(refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as mesh:
            try:
                while st.session_state.running:
                    ret,frame=cap.read()
                    if not ret:break
                    h,w=frame.shape[:2]
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    res=mesh.process(rgb)

                    ear,mr,yaw,pitch=0,0,0,0
                    face=False
                    if res.multi_face_landmarks:
                        face=True
                        lm=res.multi_face_landmarks[0].landmark
                        pts=[(int(p.x*w),int(p.y*h)) for p in lm]
                        ear=eye_aspect_ratio(pts,LEFT_EYE_IDX,RIGHT_EYE_IDX)
                        mr=mouth_open_ratio(pts)
                        try:
                            ipts=np.array([pts[i] for i in LMKS_IDX],dtype=np.float64)
                            fl=w; c=(w/2,h/2)
                            cam=np.array([[fl,0,c[0]],[0,fl,c[1]],[0,0,1]],dtype=np.float64)
                            suc,rvec,tvec=cv2.solvePnP(MODEL_POINTS,ipts,cam,np.zeros((4,1)))
                            R,_=cv2.Rodrigues(rvec)
                            roll,pitch,yaw=rotationMatrixToEulerAngles(R)
                        except:pass

                    now=time.time()
                    active_event=None

                    # --- Drowsiness ---
                    if face and ear<EAR_THRESHOLD:
                        if not st.session_state.start_times["Drowsiness"]:
                            st.session_state.start_times["Drowsiness"]=now
                        if (now-st.session_state.start_times["Drowsiness"])>=ALERT_DELAY and not st.session_state.alerted["Drowsiness"]:
                            start_alarm()
                            log_event("Drowsiness",yaw,pitch,ear,mr)
                            st.session_state.alerted["Drowsiness"]=True
                            active_event="Drowsiness"
                    else:
                        st.session_state.start_times["Drowsiness"]=None
                        st.session_state.alerted["Drowsiness"]=False

                    # --- Yawning ---
                    if face and mr>YAWN_RATIO_THRESHOLD:
                        if not st.session_state.start_times["Yawning"]:
                            st.session_state.start_times["Yawning"]=now
                        if (now-st.session_state.start_times["Yawning"])>=ALERT_DELAY and not st.session_state.alerted["Yawning"]:
                            start_alarm()
                            log_event("Yawning",yaw,pitch,ear,mr)
                            st.session_state.alerted["Yawning"]=True
                            active_event="Yawning"
                    else:
                        st.session_state.start_times["Yawning"]=None
                        st.session_state.alerted["Yawning"]=False

                    # --- Distraction ---
                    if (not face) or abs(yaw)>DISTRACTION_YAW or pitch>DISTRACTION_PITCH:
                        if not st.session_state.start_times["Distraction"]:
                            st.session_state.start_times["Distraction"]=now
                        if (now-st.session_state.start_times["Distraction"])>=ALERT_DELAY and not st.session_state.alerted["Distraction"]:
                            start_alarm()
                            log_event("Distraction",yaw,pitch,ear,mr)
                            st.session_state.alerted["Distraction"]=True
                            active_event="Distraction"
                    else:
                        st.session_state.start_times["Distraction"]=None
                        st.session_state.alerted["Distraction"]=False

                    # stop alarm if nothing detected
                    if not any(st.session_state.alerted.values()):
                        stop_alarm()

                    disp=frame.copy()
                    cv2.putText(disp,f"EAR:{ear:.2f}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                    cv2.putText(disp,f"MouthR:{mr:.2f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                    cv2.putText(disp,f"Yaw:{yaw:.1f}  Pitch:{pitch:.1f}",(10,85),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                    alerts=[k for k,v in st.session_state.alerted.items() if v]
                    for i,msg in enumerate(alerts):
                        cv2.putText(disp,f"⚠ {msg.upper()} DETECTED",(100,150+i*40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)

                    frame_slot.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB),use_column_width=True)
                    m1.metric("Drowsy","Yes" if st.session_state.alerted["Drowsiness"] else "No")
                    m2.metric("Yawning","Yes" if st.session_state.alerted["Yawning"] else "No")
                    m3.metric("Distraction","Yes" if st.session_state.alerted["Distraction"] else "No")
                    m4.metric("Time",datetime.now().strftime("%H:%M:%S"))
                    time.sleep(0.03)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                stop_alarm()
                cap.release()
                cv2.destroyAllWindows()
else:
    frame_slot.text("Click ▶ Start Detection to begin.")

st.markdown("---")
st.subheader("📜 Detection Log Preview")
if os.path.exists(LOG_FILE):
    import pandas as pd
    df=pd.read_csv(LOG_FILE)
    st.dataframe(df.tail(15),use_container_width=True)
else:
    st.info("No logs yet.")
