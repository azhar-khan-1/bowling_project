import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter
from collections import deque
from ultralytics import YOLO
import csv

# ═══════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════
yolo  = YOLO("yolov8n.pt")          # auto-download on first run (~6MB)

pose_model = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)
mp_draw = mp.solutions.drawing_utils
PoseLM  = mp.solutions.pose.PoseLandmark


# ═══════════════════════════════════════════════════════
#  EMA
# ═══════════════════════════════════════════════════════
class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.val   = None

    def update(self, v):
        self.val = v if self.val is None else self.alpha*v + (1-self.alpha)*self.val
        return self.val


# ═══════════════════════════════════════════════════════
#  Bowler Tracker  (YOLO-based, locks onto one person)
# ═══════════════════════════════════════════════════════
class BowlerTracker:
    """
    First frame  → pick the largest 'person' box as bowler.
    Next frames  → follow that track_id via YOLO ByteTrack.
    Fallback     → if track lost, re-pick largest person.
    """
    def __init__(self, pad=40):
        self.pad      = pad
        self.track_id = None
        self.ema_x1   = EMA(0.15); self.ema_y1 = EMA(0.15)
        self.ema_x2   = EMA(0.15); self.ema_y2 = EMA(0.15)
        self.last_box = None        # (x1,y1,x2,y2) full-frame

    def update(self, frame):
        results = yolo.track(frame, persist=True,
                             classes=[0],        # 0 = person
                             tracker="bytetrack.yaml",
                             verbose=False)[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return self.last_box   # use last known

        # build list of (area, track_id, x1,y1,x2,y2)
        candidates = []
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            area = (x2-x1)*(y2-y1)
            tid  = int(box.id[0]) if box.id is not None else -1
            candidates.append((area, tid, x1, y1, x2, y2))

        # lock onto track_id if seen before
        if self.track_id is not None:
            matched = [c for c in candidates if c[1] == self.track_id]
            if matched:
                _, _, x1, y1, x2, y2 = matched[0]
            else:
                # lost — re-pick largest
                self.track_id = None

        if self.track_id is None:
            candidates.sort(key=lambda c: c[0], reverse=True)
            area, tid, x1, y1, x2, y2 = candidates[0]
            self.track_id = tid

        H, W = frame.shape[:2]
        pad   = self.pad
        sx1 = int(self.ema_x1.update(max(0,   x1 - pad)))
        sy1 = int(self.ema_y1.update(max(0,   y1 - pad)))
        sx2 = int(self.ema_x2.update(min(W-1, x2 + pad)))
        sy2 = int(self.ema_y2.update(min(H-1, y2 + pad)))

        self.last_box = (sx1, sy1, sx2, sy2)
        return self.last_box


# ═══════════════════════════════════════════════════════
#  Phase Detector
# ═══════════════════════════════════════════════════════
class PhaseDetector:
    PHASES = ["RUNUP", "LOAD", "DELIVERY", "FOLLOWTHROUGH"]
    COLORS_BGR = {
        "RUNUP":         (0,   200, 255),
        "LOAD":          (0,   140, 255),
        "DELIVERY":      (0,   0,   255),
        "FOLLOWTHROUGH": (0,   220, 80),
    }
    COLORS_MPL = {
        "RUNUP":         "#FFD700",
        "LOAD":          "#FF8C00",
        "DELIVERY":      "#FF3333",
        "FOLLOWTHROUGH": "#33CC66",
    }
    LABELS = {
        "RUNUP":         "Run-Up",
        "LOAD":          "Load / Coil",
        "DELIVERY":      "Delivery",
        "FOLLOWTHROUGH": "Follow-Through",
    }

    def __init__(self, fps):
        self.fps          = fps
        self.phase        = "RUNUP"
        self.phase_frames = 0
        self.history      = []
        self.phase_log    = []
        self._start       = 0
        self.frame_no     = 0
        self._kbuf = deque(maxlen=9)
        self._tbuf = deque(maxlen=9)
        self._abuf = deque(maxlen=9)

    def _avg(self, buf): return sum(buf)/len(buf) if buf else 0

    def update(self, knee, trunk, arm):
        self._kbuf.append(knee)
        self._tbuf.append(trunk)
        self._abuf.append(arm)
        self.frame_no += 1
        k, t, a = self._avg(self._kbuf), self._avg(self._tbuf), self._avg(self._abuf)
        old = self.phase

        if   self.phase == "RUNUP"   and t < 155 and k < 160:
            self.phase = "LOAD"
        elif self.phase == "LOAD"    and a > 140:
            self.phase = "DELIVERY"
        elif self.phase == "DELIVERY" and a < 110 and self.phase_frames > self.fps*0.3:
            self.phase = "FOLLOWTHROUGH"

        if self.phase != old:
            self.phase_log.append((old, self._start, self.frame_no-1))
            self._start       = self.frame_no
            self.phase_frames = 0
        else:
            self.phase_frames += 1

        self.history.append(self.phase)
        return self.phase

    def finalize(self):
        self.phase_log.append((self.phase, self._start, self.frame_no))

    def color(self):  return self.COLORS_BGR[self.phase]
    def label(self):  return self.LABELS[self.phase]


# ═══════════════════════════════════════════════════════
#  Drawing helpers
# ═══════════════════════════════════════════════════════
def calc_angle(a, b, c):
    a,b,c = map(np.array, (a,b,c))
    ba,bc = a-b, c-b
    cos   = np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8),-1,1)
    return np.degrees(np.arccos(cos))


def draw_angle_arc(frame, vertex, p1, p2, angle, color, radius=32):
    v  = np.array(vertex, dtype=float)
    d1 = np.array(p1,dtype=float)-v
    d2 = np.array(p2,dtype=float)-v
    if np.linalg.norm(d1)<1 or np.linalg.norm(d2)<1: return
    a1 = int(np.degrees(np.arctan2(d1[1],d1[0])))
    a2 = int(np.degrees(np.arctan2(d2[1],d2[0])))
    cv2.ellipse(frame, tuple(map(int,v)), (radius,radius), 0, a1, a2, color, 2)
    mid = np.radians((a1+a2)/2)
    tx  = int(v[0]+(radius+14)*np.cos(mid))
    ty  = int(v[1]+(radius+14)*np.sin(mid))
    cv2.putText(frame, f"{int(angle)}", (tx,ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_hud(frame, pd, knee, arm, trunk, frame_no, fps):
    ph_col = pd.color()
    H, W   = frame.shape[:2]

    # top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (W,40), (20,20,20), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (0,0), (230,40), ph_col, -1)
    cv2.putText(frame, pd.label(), (8,28),
                cv2.FONT_HERSHEY_DUPLEX, 0.78, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame {frame_no:04d}  {frame_no/fps:.2f}s",
                (238,27), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200,200,200), 1, cv2.LINE_AA)

    # angle panel
    for i,(name,val,col) in enumerate([
            ("Knee",  knee,  (0,220,80)),
            ("Arm",   arm,   (80,80,255)),
            ("Trunk", trunk, (255,80,80))]):
        y = 75 + i*40
        cv2.putText(frame, f"{name}:", (12,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(val):>3} deg", (88,y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 2, cv2.LINE_AA)

    # phase timeline
    bar_y = H - 20
    cv2.rectangle(frame,(0,bar_y),(W,H),(30,30,30),-1)
    total = max(frame_no,1)
    seg   = W / total
    for i,ph in enumerate(pd.history):
        x1 = int(i*seg); x2 = int((i+1)*seg)
        cv2.rectangle(frame,(x1,bar_y),(x2,H),PhaseDetector.COLORS_BGR[ph],-1)
    cx = int((frame_no/total)*W)
    cv2.line(frame,(cx,bar_y-3),(cx,H),(255,255,255),2)


def savgol_smooth(data, w=13, p=3):
    if len(data)<w: return list(data)
    w = w|1; w = min(w, len(data)|1)
    return savgol_filter(data,w,p).tolist()


# ═══════════════════════════════════════════════════════
#  Video I/O
# ═══════════════════════════════════════════════════════
cap = cv2.VideoCapture("bowling.mp4")
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter("output.mp4", fourcc, fps, (W,H))

tracker      = BowlerTracker(pad=45)
phase_det    = PhaseDetector(fps)
ema_com_x    = EMA(0.25); ema_com_y = EMA(0.25)
com_trail    = deque(maxlen=30)

COM_LM = [
    (PoseLM.LEFT_SHOULDER,  1.5),(PoseLM.RIGHT_SHOULDER, 1.5),
    (PoseLM.LEFT_HIP,       2.0),(PoseLM.RIGHT_HIP,      2.0),
    (PoseLM.LEFT_KNEE,      1.0),(PoseLM.RIGHT_KNEE,     1.0),
    (PoseLM.LEFT_ANKLE,     0.5),(PoseLM.RIGHT_ANKLE,    0.5),
]

knee_angles=[]; arm_angles=[]; trunk_angles=[]
angular_velocity=[]; com_x_raw=[]; com_y_raw=[]; phases_raw=[]
prev_arm = None; frame_no = 0


# ═══════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════
print("\nProcessing video... (ESC to stop early)\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_no += 1

    # ── YOLO: get stable bowler box ──────────────────
    box = tracker.update(frame)

    if box is None:
        out.write(frame)
        cv2.imshow("Bowling Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        continue

    bx1, by1, bx2, by2 = box
    rw = bx2 - bx1; rh = by2 - by1
    if rw < 10 or rh < 10:
        out.write(frame); continue

    # ── MediaPipe on cropped bowler ──────────────────
    crop    = frame[by1:by2, bx1:bx2].copy()
    rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    ph_color = phase_det.color()

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # skeleton drawn on crop then pasted back
        dot_spec  = mp_draw.DrawingSpec(color=(255,255,255), thickness=-1, circle_radius=5)
        conn_spec = mp_draw.DrawingSpec(color=ph_color,      thickness=3)
        mp_draw.draw_landmarks(crop, results.pose_landmarks,
                               mp.solutions.pose.POSE_CONNECTIONS,
                               dot_spec, conn_spec)
        frame[by1:by2, bx1:bx2] = crop

        # full-frame landmark coords
        def pt(lm_e):
            l = lm[lm_e.value]
            return [bx1 + l.x*rw, by1 + l.y*rh]

        shoulder = pt(PoseLM.RIGHT_SHOULDER)
        elbow    = pt(PoseLM.RIGHT_ELBOW)
        wrist    = pt(PoseLM.RIGHT_WRIST)
        hip      = pt(PoseLM.RIGHT_HIP)
        knee     = pt(PoseLM.RIGHT_KNEE)
        ankle    = pt(PoseLM.RIGHT_ANKLE)

        knee_a  = calc_angle(hip,      knee,  ankle)
        arm_a   = calc_angle(shoulder, elbow, wrist)
        trunk_a = calc_angle(shoulder, hip,   knee)

        knee_angles.append(knee_a)
        arm_angles.append(arm_a)
        trunk_angles.append(trunk_a)
        angular_velocity.append((arm_a-prev_arm)*fps if prev_arm else 0)
        prev_arm = arm_a

        phase = phase_det.update(knee_a, trunk_a, arm_a)
        phases_raw.append(phase)

        # angle arcs
        draw_angle_arc(frame, knee,  hip,      ankle, knee_a,  (0,220,80))
        draw_angle_arc(frame, elbow, shoulder, wrist, arm_a,   (80,80,255))
        draw_angle_arc(frame, hip,   shoulder, knee,  trunk_a, (255,80,80))

        # weighted COM
        wx, wy, wt = 0, 0, 0
        for (lm_e, w) in COM_LM:
            l = lm[lm_e.value]
            if l.visibility > 0.2:
                wx += (bx1 + l.x*rw)*w
                wy += (by1 + l.y*rh)*w
                wt += w
        raw_cx = wx/wt if wt>0 else (bx1+bx2)/2
        raw_cy = wy/wt if wt>0 else (by1+by2)/2

        com_x = int(ema_com_x.update(raw_cx))
        com_y = int(ema_com_y.update(raw_cy))
        com_x_raw.append(com_x); com_y_raw.append(com_y)
        com_trail.append((com_x, com_y))

        for i in range(1, len(com_trail)):
            cv2.line(frame, com_trail[i-1], com_trail[i], (255,0,255), 2)
        cv2.circle(frame, (com_x,com_y), 9, (255,0,255), -1)
        cv2.circle(frame, (com_x,com_y), 9, (255,255,255), 2)
        cv2.putText(frame, "COM", (com_x+11, com_y-11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)

        draw_hud(frame, phase_det, knee_a, arm_a, trunk_a, frame_no, fps)

    # ── YOLO box drawn LAST so always visible on top ──
    # thick outer border + thin inner
    cv2.rectangle(frame, (bx1-2, by1-2), (bx2+2, by2+2), (0,0,0),    3)  # black shadow
    cv2.rectangle(frame, (bx1,   by1),   (bx2,   by2),   ph_color,    3)  # phase color
    # corner accents
    corner_len = 20
    corner_t   = 4
    for cx, cy, dx, dy in [
        (bx1, by1,  1,  1), (bx2, by1, -1,  1),
        (bx1, by2,  1, -1), (bx2, by2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy),            (255,255,255), corner_t)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len),            (255,255,255), corner_t)

    # label tag above box
    tag   = f" BOWLER  ID:{tracker.track_id} "
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    tag_y = max(by1 - 8, th + 4)
    cv2.rectangle(frame, (bx1, tag_y - th - 4), (bx1 + tw, tag_y + 4), ph_color, -1)
    cv2.putText(frame, tag, (bx1, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("Bowling Analyzer  [ESC=quit]", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

phase_det.finalize()
cap.release(); out.release(); cv2.destroyAllWindows()
print("Video saved → output.mp4")


# ═══════════════════════════════════════════════════════
#  CSV Export
# ═══════════════════════════════════════════════════════
n    = len(knee_angles)
time = [i/fps for i in range(n)]

with open("biomechanics_data.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["Frame","Time_s","Phase","Knee_deg","Arm_deg","Trunk_deg",
                "Arm_AngVel_deg_s","COM_X_px","COM_Y_px"])
    for i in range(n):
        w.writerow([i+1, round(time[i],3),
                    phases_raw[i] if i<len(phases_raw) else "",
                    round(knee_angles[i],2), round(arm_angles[i],2),
                    round(trunk_angles[i],2), round(angular_velocity[i],2),
                    com_x_raw[i] if i<len(com_x_raw) else "",
                    com_y_raw[i] if i<len(com_y_raw) else ""])
print("CSV saved → biomechanics_data.csv")


# ── Phase summary ──
print("\n── Phase Summary ─────────────────────────────")
for ph,sf,ef in phase_det.phase_log:
    print(f"  {PhaseDetector.LABELS[ph]:<20}  "
          f"frames {sf:>4}–{ef:>4}  ({(ef-sf)/fps:.2f}s)")
print("──────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════
#  Graphs
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(15,11))
fig.suptitle("Bowling Biomechanics — YOLO+MediaPipe",
             fontsize=15, fontweight='bold')

def shade(ax):
    for ph,sf,ef in phase_det.phase_log:
        ax.axvspan(sf/fps, ef/fps,
                   color=PhaseDetector.COLORS_MPL[ph], alpha=0.18)

datasets = [
    (savgol_smooth(knee_angles),      "Knee Angle (°)",             "tab:green"),
    (savgol_smooth(arm_angles),       "Arm Angle (°)",              "tab:red"),
    (savgol_smooth(trunk_angles),     "Trunk Angle (°)",            "tab:blue"),
    (savgol_smooth(angular_velocity), "Arm Angular Velocity (°/s)", "tab:orange"),
    (savgol_smooth(com_x_raw),        "COM Horizontal (px)",        "tab:purple"),
    (savgol_smooth(com_y_raw),        "COM Vertical (px)",          "tab:brown"),
]

for ax,(data,title,color) in zip(axes.flat, datasets):
    shade(ax)
    ax.plot(time[:len(data)], data, color=color, linewidth=2)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines[['top','right']].set_visible(False)

patches = [mpatches.Patch(color=PhaseDetector.COLORS_MPL[ph],
                           label=PhaseDetector.LABELS[ph], alpha=0.8)
           for ph in PhaseDetector.PHASES]
fig.legend(handles=patches, loc='lower center', ncol=4,
           fontsize=10, frameon=True, title="Bowling Phase")

plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig("biomechanics_graphs.png", dpi=160)
plt.show()
print("Graphs saved → biomechanics_graphs.png")
