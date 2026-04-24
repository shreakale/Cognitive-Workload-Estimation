import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import datetime
from collections import deque, Counter

st.set_page_config(
    page_title="Cognitive Load Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .state-relaxed  { color:#00ff88;font-size:42px;font-weight:bold;text-align:center;text-shadow:0 0 20px #00ff88;font-family:monospace; }
    .state-focused  { color:#ff8c00;font-size:42px;font-weight:bold;text-align:center;text-shadow:0 0 20px #ff8c00;font-family:monospace; }
    .state-confused { color:#ff4444;font-size:42px;font-weight:bold;text-align:center;text-shadow:0 0 20px #ff4444;font-family:monospace; }
    .state-analyzing{ color:#888888;font-size:32px;font-weight:bold;text-align:center;font-family:monospace; }
    .metric-value   { font-size:26px;font-weight:bold;color:#00ff88;font-family:monospace; }
    .metric-label   { font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)



class Config:
    CAMERA_WIDTH        = 640
    CAMERA_HEIGHT       = 480
    FACE_SCALE          = 1.1
    FACE_MIN_NEIGHBORS  = 5
    EYE_SCALE           = 1.1
    EYE_MIN_NEIGHBORS   = 10
    EAR_BLINK_THRESH    = 0.26   # eye closing threshold
    BLINK_CONSEC_FRAMES = 3      # frames below threshold = blink
    PREDICTION_INTERVAL = 3.0    # seconds between state updates
    WARMUP_SECONDS      = 8.0    # baseline collection before classifying



class BlinkDetector:
    def __init__(self):
        self.blink_count      = 0
        self.blink_start_time = None
        self.blink_durations  = []
        self.is_blinking      = False
        self.last_blink_time  = 0
        self.consec_frames    = 0
        self.session_start    = time.time()

    def update(self, ear):
        current_time = time.time()
        features = {'ear': ear, 'blink_rate': 0.0, 'blink_duration': 0.0}

        if ear < Config.EAR_BLINK_THRESH:
            self.consec_frames += 1
            if self.consec_frames >= Config.BLINK_CONSEC_FRAMES and not self.is_blinking:
                self.is_blinking      = True
                self.blink_start_time = current_time
        else:
            if self.is_blinking:
                self.is_blinking = False
                if self.blink_start_time:
                    duration = current_time - self.blink_start_time
                    self.blink_durations.append(duration)
                    self.blink_count    += 1
                    self.last_blink_time = current_time
                    features['blink_duration'] = duration * 1000
            self.consec_frames = 0

        # blinks per minute over whole session
        session_elapsed = max(current_time - self.session_start, 1)
        features['blink_rate'] = (self.blink_count / session_elapsed) * 60
        return features


# ─────────────────────────────────────────────
#  COGNITIVE STATE CLASSIFIER
#
#  Research basis:
#   • Normal blink rate: 12–20 /min (Bentivoglio et al. 1997)
#   • High cognitive load → blink suppression (8–12 /min) + EAR ↓ (gaze lock)
#   • Mental fatigue / confusion → slow/long blinks (>200 ms), irregular rate
#   • Relaxed → normal blink rate, stable EAR ~0.28–0.35
#
#  Method: evidence scoring across 4 features → majority-vote smoothing
# ─────────────────────────────────────────────
class CognitiveClassifier:
    def __init__(self):
        self.ear_history       = deque(maxlen=90)    # ~3 s at 30 fps
        self._smoothing_window = deque(maxlen=5)     # majority-vote buffer

    def update_history(self, ear, blink_rate):
        self.ear_history.append(ear)

    def classify(self, features, session_elapsed):
        """Returns (state_str, confidence_pct, prob_dict)"""
        blink_rate     = features.get('blink_rate',     0.0)
        blink_duration = features.get('blink_duration', 0.0)
        ear            = features.get('ear',            0.28)
        no_face        = features.get('no_face',        False)

        if no_face or session_elapsed < Config.WARMUP_SECONDS:
            return 'Analyzing...', 0.0, {'Relaxed': 0.33, 'Focused': 0.34, 'Confused': 0.33}

        scores = {'Relaxed': 0.0, 'Focused': 0.0, 'Confused': 0.0}

        # ── 1. BLINK RATE ──────────────────────────────────────────
        if blink_rate == 0:
            scores['Confused'] += 1.5
        elif blink_rate < 6:
            scores['Focused']  += 2.0
            scores['Confused'] += 1.0
        elif blink_rate < 12:
            scores['Focused']  += 3.0
        elif blink_rate <= 20:
            scores['Relaxed']  += 3.0
            scores['Focused']  += 1.0
        else:
            scores['Confused'] += 2.5
            scores['Relaxed']  += 0.5

        # ── 2. EAR (rolling mean for stability) ────────────────────
        mean_ear = float(np.mean(self.ear_history)) if self.ear_history else ear
        if mean_ear < 0.22:
            scores['Confused'] += 2.5
        elif mean_ear < 0.27:
            scores['Focused']  += 2.0
        elif mean_ear < 0.33:
            scores['Relaxed']  += 2.0
            scores['Focused']  += 1.0
        else:
            scores['Focused']  += 1.5
            scores['Confused'] += 0.5

        # ── 3. BLINK DURATION ──────────────────────────────────────
        if blink_duration > 0:
            if blink_duration > 300:
                scores['Confused'] += 2.0
            elif blink_duration > 180:
                scores['Confused'] += 1.0
                scores['Focused']  += 0.5
            elif blink_duration < 80:
                scores['Relaxed']  += 1.5
            else:
                scores['Relaxed']  += 1.0
                scores['Focused']  += 1.0

        # ── 4. EAR VARIANCE (gaze stability) ───────────────────────
        if len(self.ear_history) > 10:
            ear_var = float(np.var(self.ear_history))
            if ear_var < 0.0005:
                scores['Focused']  += 1.5
            elif ear_var > 0.003:
                scores['Confused'] += 1.5
            else:
                scores['Relaxed']  += 0.5

        # ── Normalise to probabilities ──────────────────────────────
        total = sum(scores.values()) or 1.0
        probs = {k: v / total for k, v in scores.items()}

        # ── Majority-vote smoothing ─────────────────────────────────
        raw_winner = max(scores, key=scores.get)
        self._smoothing_window.append(raw_winner)
        smoothed   = Counter(self._smoothing_window).most_common(1)[0][0]
        confidence = probs[smoothed] * 100

        print(f"[SCORES] R={scores['Relaxed']:.2f} F={scores['Focused']:.2f} "
              f"C={scores['Confused']:.2f} | blink={blink_rate:.1f}/min "
              f"ear={mean_ear:.3f} dur={blink_duration:.0f}ms → {smoothed} ({confidence:.1f}%)")

        return smoothed, confidence, probs



def export_session(state_history, state_times, start_time):
    os.makedirs("output", exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dur = time.time() - start_time
    df  = pd.DataFrame({'timestamp': list(range(len(state_history))), 'state': list(state_history)})
    path = f"output/session_{ts}.csv"
    df.to_csv(path, index=False)
    dominant = max(state_times, key=state_times.get)
    print(f"\n{'='*50}\nSESSION SUMMARY\n{'='*50}")
    print(f"Duration: {dur:.1f}s | Dominant: {dominant}")
    for k, v in state_times.items():
        print(f"  {k}: {v:.0f}s")
    print(f"Saved: {path}\n{'='*50}")



def main():
    st.markdown("""
    <h1 style='text-align:center;color:#e6edf3;font-family:monospace;letter-spacing:3px;'>
    COGNITIVE LOAD MONITOR
    </h1>
    <p style='text-align:center;color:#8b949e;font-family:monospace;'>
    Real-time Mental State Detection — Camera-Native Classifier
    </p>
    <hr style='border-color:#30363d;'>
    """, unsafe_allow_html=True)

    defaults = {
        'running'      : False,
        'start_time'   : time.time(),
        'blink_det'    : None,
        'classifier'   : None,
        'blink_history': deque(maxlen=60),
        'state_history': deque(maxlen=30),
        'current_state': 'Analyzing...',
        'current_conf' : 0.0,
        'current_probs': {'Relaxed': 0.33, 'Focused': 0.34, 'Confused': 0.33},
        'last_pred'    : 0.0,
        'state_times'  : {'Relaxed': 0, 'Focused': 0, 'Confused': 0},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 START SESSION", use_container_width=True, type="primary"):
            st.session_state.running       = True
            st.session_state.start_time    = time.time()
            st.session_state.blink_det     = BlinkDetector()
            st.session_state.classifier    = CognitiveClassifier()
            st.session_state.blink_history = deque(maxlen=60)
            st.session_state.state_history = deque(maxlen=30)
            st.session_state.state_times   = {'Relaxed': 0, 'Focused': 0, 'Confused': 0}
            st.session_state.current_state = 'Analyzing...'
            st.session_state.last_pred     = time.time()
    with c2:
        if st.button("⏹️ STOP SESSION", use_container_width=True):
            st.session_state.running = False
            export_session(st.session_state.state_history,
                           st.session_state.state_times,
                           st.session_state.start_time)
    with c3:
        if st.button("🔄 RESET", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 📹 Live Camera Feed")
        cam_ph      = st.empty()
        st.markdown("#### 📊 Blink Rate Over Time")
        chart_ph    = st.empty()
        st.markdown("#### 📈 Session Summary")
        summary_ph  = st.empty()
    with right:
        st.markdown("#### 🧠 Cognitive State")
        state_ph    = st.empty()
        st.markdown("#### 📋 Class Probabilities")
        prob_ph     = st.empty()
        st.markdown("#### 📏 Biometric Metrics")
        metrics_ph  = st.empty()
        st.markdown("#### ⏳ State Timeline")
        timeline_ph = st.empty()

    warmup_ph = st.empty()

   
    if st.session_state.running:
        face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cas  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)

        if not cap.isOpened():
            st.error("❌ Camera not found!")
            st.session_state.running = False
            return

        blink_det  = st.session_state.blink_det
        classifier = st.session_state.classifier

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Camera read error!")
                break

            frame = cv2.flip(frame, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cas.detectMultiScale(
                gray,
                scaleFactor=Config.FACE_SCALE,
                minNeighbors=Config.FACE_MIN_NEIGHBORS,
                minSize=(60, 60),
            )

            ear           = 0.28
            face_detected = len(faces) > 0
            features      = {}

            for (fx, fy, fw, fh) in faces[:1]:
                color = {'Relaxed':(0,255,136),'Focused':(0,140,255),'Confused':(68,68,255)}.get(
                    st.session_state.current_state, (200,200,200))
                cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), color, 3)

                roi_gray  = gray[fy:fy+fh, fx:fx+fw]
                roi_color = frame[fy:fy+fh, fx:fx+fw]

                eyes = eye_cas.detectMultiScale(
                    roi_gray,
                    scaleFactor=Config.EYE_SCALE,
                    minNeighbors=Config.EYE_MIN_NEIGHBORS,
                    minSize=(20, 20),
                )

                if len(eyes) >= 2:
                    for (ex,ey,ew,eh) in eyes[:2]:
                        cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    ear = float(np.mean([e[3]/max(e[2],1) for e in eyes[:2]]))

            blink_feats = blink_det.update(ear)
            features.update(blink_feats)
            features['no_face'] = not face_detected

            classifier.update_history(ear, features['blink_rate'])
            st.session_state.blink_history.append(features['blink_rate'])

            session_elapsed = time.time() - st.session_state.start_time
            remaining = max(0, Config.WARMUP_SECONDS - session_elapsed)
            if remaining > 0:
                warmup_ph.info(f"⏳ Calibrating baseline… {remaining:.1f}s remaining — look naturally at the screen")
            else:
                warmup_ph.empty()

            if time.time() - st.session_state.last_pred >= Config.PREDICTION_INTERVAL:
                state, conf, probs = classifier.classify(features, session_elapsed)
                st.session_state.current_state = state
                st.session_state.current_conf  = conf
                st.session_state.current_probs = probs
                st.session_state.last_pred     = time.time()
                if state in st.session_state.state_times:
                    st.session_state.state_history.append(state)
                    st.session_state.state_times[state] += Config.PREDICTION_INTERVAL

            if not face_detected:
                cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # ── Camera ──
            cam_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # ── State ──
            s    = st.session_state.current_state
            conf = st.session_state.current_conf
            css  = {'Relaxed':'state-relaxed','Focused':'state-focused','Confused':'state-confused'}.get(s,'state-analyzing')
            state_ph.markdown(f"""
            <div class='{css}'>[ {s.upper()} ]</div>
            <p style='text-align:center;color:#8b949e;font-family:monospace;font-size:16px;'>Confidence: {conf:.1f}%</p>
            """, unsafe_allow_html=True)

            # ── Probability bars ──
            probs  = st.session_state.current_probs
            labels = ['Relaxed','Focused','Confused']
            colors = ['#00ff88','#ff8c00','#ff4444']
            bgs    = ['#1e2a1e','#2a1e0a','#2a0a0a']
            bars   = ""
            for lbl, col, bg in zip(labels, colors, bgs):
                pct = probs.get(lbl, 0.0) * 100
                bars += f"""
                <div style='margin:8px 0;'>
                    <span style='color:{col};font-family:monospace;font-size:13px;'>{lbl.upper()}</span>
                    <div style='background:{bg};border-radius:4px;height:18px;margin-top:3px;'>
                        <div style='background:{col};width:{pct:.1f}%;height:100%;border-radius:4px;transition:width 0.5s;'></div>
                    </div>
                    <span style='color:#8b949e;font-size:11px;'>{pct:.1f}%</span>
                </div>"""
            prob_ph.markdown(f"<div style='padding:5px;'>{bars}</div>", unsafe_allow_html=True)

            # ── Metrics ──
            elapsed = int(time.time() - st.session_state.start_time)
            mins, secs  = divmod(elapsed, 60)
            blink_count = blink_det.blink_count
            blink_rate  = features.get('blink_rate', 0.0)
            avg_dur     = (float(np.mean(blink_det.blink_durations[-20:]))*1000
                           if blink_det.blink_durations else 0.0)
            metrics_ph.markdown(f"""
            <div style='display:flex;gap:8px;flex-wrap:wrap;'>
                <div class='metric-card'><div class='metric-value'>{blink_count}</div><div class='metric-label'>Blinks</div></div>
                <div class='metric-card'><div class='metric-value'>{blink_rate:.1f}</div><div class='metric-label'>Rate/min</div></div>
                <div class='metric-card'><div class='metric-value'>{ear:.2f}</div><div class='metric-label'>EAR</div></div>
                <div class='metric-card'><div class='metric-value'>{avg_dur:.0f}ms</div><div class='metric-label'>Avg Blink</div></div>
                <div class='metric-card'><div class='metric-value'>{mins:02d}:{secs:02d}</div><div class='metric-label'>Time</div></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Timeline ──
            history = list(st.session_state.state_history)
            STATE_COLORS = {'Relaxed':'#00ff88','Focused':'#ff8c00','Confused':'#ff4444'}
            fallback = '#888'
            dots = "".join(
                f"<span title='{s}' style='display:inline-block;width:18px;height:18px;"
                f"border-radius:50%;background:{STATE_COLORS.get(s, fallback)};margin:2px;'></span>"
                for s in history
            )
            timeline_ph.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:10px;min-height:40px;'>{dots}</div>",
                unsafe_allow_html=True)

            # ── Blink chart ──
            bh = list(st.session_state.blink_history)
            if bh:
                chart_ph.line_chart(pd.DataFrame({'Blink Rate (per min)': bh}),
                                    use_container_width=True, height=150)

            # ── Summary ──
            st_times = st.session_state.state_times
            total    = sum(st_times.values()) or 1
            summary_ph.markdown(f"""
            <div style='display:flex;gap:10px;'>
                <div class='metric-card' style='flex:1;'>
                    <div style='color:#00ff88;font-size:18px;font-weight:bold;'>{st_times['Relaxed']:.0f}s</div>
                    <div class='metric-label'>Relaxed</div>
                    <div style='background:#1e2a1e;border-radius:4px;height:6px;margin-top:6px;'>
                        <div style='background:#00ff88;width:{st_times["Relaxed"]/total*100:.0f}%;height:100%;border-radius:4px;'></div>
                    </div>
                </div>
                <div class='metric-card' style='flex:1;'>
                    <div style='color:#ff8c00;font-size:18px;font-weight:bold;'>{st_times['Focused']:.0f}s</div>
                    <div class='metric-label'>Focused</div>
                    <div style='background:#2a1e0a;border-radius:4px;height:6px;margin-top:6px;'>
                        <div style='background:#ff8c00;width:{st_times["Focused"]/total*100:.0f}%;height:100%;border-radius:4px;'></div>
                    </div>
                </div>
                <div class='metric-card' style='flex:1;'>
                    <div style='color:#ff4444;font-size:18px;font-weight:bold;'>{st_times['Confused']:.0f}s</div>
                    <div class='metric-label'>Confused</div>
                    <div style='background:#2a0a0a;border-radius:4px;height:6px;margin-top:6px;'>
                        <div style='background:#ff4444;width:{st_times["Confused"]/total*100:.0f}%;height:100%;border-radius:4px;'></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.03)

        cap.release()

    else:
        cam_ph = st.empty()
        cam_ph.markdown("""
        <div style='background:#161b22;border-radius:12px;height:300px;
        display:flex;align-items:center;justify-content:center;border:2px dashed #30363d;'>
            <p style='color:#8b949e;font-family:monospace;font-size:18px;'>
            Press 🚀 START SESSION to begin monitoring
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()