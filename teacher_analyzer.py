"""
teacher_analyzer.py
=====================
Combined Teacher Analysis System
Selects ONE random video from the 'teacher' folder and runs:
  1. Teaching Mode Detection  (boardonly / pptonly / boardandppt)
  2. Enthusiasm Detection     (enthusiastic / not enthusiastic %)

No retraining needed — loads pre-trained models directly.

Folder structure expected:
    teacher/
      *.mp4  (or .avi / .mov / .mkv)
      best_model.pth          ← teaching mode model
      enthusiasm_lstm.pt      ← enthusiasm model
      feature_scaler.pkl      ← enthusiasm scaler

Output:
    teacher/outputs/
      combined_output_<video>_<timestamp>.mp4   (annotated video)
      combined_report_<video>_<timestamp>.json  (full report)

Keys while playing:
  Q / ESC  → quit
  S        → save screenshot
  P        → pause / resume

Run:
  python teacher_analyzer.py
  python teacher_analyzer.py --folder path/to/teacher
"""

import argparse
import os
import time
import random
import json
import math
import warnings
import logging
from datetime import datetime
from pathlib import Path
from collections import deque, Counter
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ══════════════════════════════════════════════════════════════
#  HARDCODED DEFAULT PATH — change if needed
# ══════════════════════════════════════════════════════════════
DEFAULT_TEACHER_FOLDER = r"teacher"
# ══════════════════════════════════════════════════════════════

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART A : TEACHING MODE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class TeachingClassifier(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        base          = models.mobilenet_v2(weights=None)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(1280, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512,  128), nn.ReLU(),
            nn.Linear(128, n),
        )
    def forward(self, x):
        return self.head(self.pool(self.features(x)).view(x.size(0), -1))


class TeachingPredictor:
    def __init__(self, model_path: str):
        ckpt             = torch.load(model_path, map_location=DEVICE)
        self.class_names = ckpt.get("class_names", ["boardonly", "pptonly", "boardandppt"])
        img_size         = ckpt.get("img_size", 224)
        self.model       = TeachingClassifier(ckpt.get("num_classes", 3)).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.buf = deque(maxlen=5)

    def predict(self, frame_bgr):
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.tfm(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        idx    = int(np.argmax(probs))
        raw    = self.class_names[idx]
        conf   = float(probs[idx])
        self.buf.append(raw)
        smooth = Counter(self.buf).most_common(1)[0][0]
        return raw, smooth, conf, probs


# ═══════════════════════════════════════════════════════════════════════════════
#  PART B : ENTHUSIASM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    MP_POSE = mp.solutions.pose
    LANDMARKS = MP_POSE.PoseLandmark
    LEFT_SHOULDER  = LANDMARKS.LEFT_SHOULDER.value
    RIGHT_SHOULDER = LANDMARKS.RIGHT_SHOULDER.value
    LEFT_WRIST     = LANDMARKS.LEFT_WRIST.value
    RIGHT_WRIST    = LANDMARKS.RIGHT_WRIST.value
    LEFT_EAR       = LANDMARKS.LEFT_EAR.value
    RIGHT_EAR      = LANDMARKS.RIGHT_EAR.value
    LEFT_HIP       = LANDMARKS.LEFT_HIP.value
    RIGHT_HIP      = LANDMARKS.RIGHT_HIP.value
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("  [WARNING] mediapipe not installed — enthusiasm detection will use motion-only mode.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


ENTHU_CFG = {
    "seq_len": 16,
    "stride": 4,
    "n_features": 15,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.4,
    "threshold": 0.45,
    "target_fps": 6,
}


class EnthusiasmLSTM(nn.Module):
    def __init__(self, n_features=15, hidden=128, layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden * 2)
        self.dropout    = nn.Dropout(0.3)
        self.attn       = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(out), dim=1)
        out    = torch.sum(out * attn_w, dim=1)
        out    = self.dropout(out)
        return self.classifier(out)


def _lm(results, idx: int) -> Optional[np.ndarray]:
    if results is None or results.pose_landmarks is None:
        return None
    lm = results.pose_landmarks.landmark[idx]
    if lm.visibility < 0.3:
        return None
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def detect_emotion_lightweight(frame_bgr):
    gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0
    happy      = brightness
    surprise   = brightness * 0.5
    neutral    = 1 - brightness
    sad        = (1 - brightness) * 0.5
    angry      = 0.1
    fear       = 0.1
    vec        = np.array([happy, surprise, neutral, sad, angry, fear], dtype=np.float32)
    return vec, float(max(vec))


def extract_pose_features_fn(frame_bgr, pose_model, prev_lw, prev_rw):
    if not MEDIAPIPE_AVAILABLE or pose_model is None:
        return 0.0, 0.0, 1.0, 0.0, 0.0, None, None

    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    left_vel = right_vel = shoulder_open = head_tilt = body_motion = 0.0
    new_lw = new_rw = None

    lsh  = _lm(results, LEFT_SHOULDER)
    rsh  = _lm(results, RIGHT_SHOULDER)
    lw   = _lm(results, LEFT_WRIST)
    rw   = _lm(results, RIGHT_WRIST)
    lear = _lm(results, LEFT_EAR)
    rear = _lm(results, RIGHT_EAR)
    lhip = _lm(results, LEFT_HIP)
    rhip = _lm(results, RIGHT_HIP)

    if lw is not None:
        new_lw = lw[:2]
        if prev_lw is not None:
            left_vel = np.linalg.norm(lw[:2] - prev_lw)
    if rw is not None:
        new_rw = rw[:2]
        if prev_rw is not None:
            right_vel = np.linalg.norm(rw[:2] - prev_rw)

    if lsh is not None and rsh is not None:
        sh_width = np.linalg.norm(lsh[:2] - rsh[:2])
        if lhip is not None and rhip is not None:
            torso_h  = np.linalg.norm(
                (lsh[:2] + rsh[:2]) / 2 - (lhip[:2] + rhip[:2]) / 2
            )
            shoulder_open = sh_width / (torso_h + 1e-6)

    if lear is not None and rear is not None:
        dx        = lear[0] - rear[0]
        dy        = lear[1] - rear[1]
        angle     = abs(math.atan2(dy, dx))
        head_tilt = min(angle / math.pi, 1.0)

    if lsh is not None and rsh is not None and lhip is not None and rhip is not None:
        torso_centre = (lsh[:2] + rsh[:2] + lhip[:2] + rhip[:2]) / 4
        if hasattr(extract_pose_features_fn, "prev_torso"):
            body_motion = np.linalg.norm(torso_centre - extract_pose_features_fn.prev_torso)
        extract_pose_features_fn.prev_torso = torso_centre

    return left_vel, right_vel, shoulder_open, head_tilt, body_motion, new_lw, new_rw


def build_frame_features_fn(frame_bgr, pose_model, prev_lw, prev_rw):
    emotion_vec, face_conf = detect_emotion_lightweight(frame_bgr)
    lv, rv, sh, ht, bm, new_lw, new_rw = extract_pose_features_fn(
        frame_bgr, pose_model, prev_lw, prev_rw
    )
    lv = min(lv * 25, 1.0)
    rv = min(rv * 25, 1.0)
    bm = min(bm * 15, 1.0)
    sh = min(sh, 2.0)
    ht = min(ht, 1.0)

    low_motion_flag  = 1.0 if (lv < 0.01 and rv < 0.01 and bm < 0.02) else 0.0
    non_expressive   = 1.0 if (emotion_vec[2] > 0.6 or emotion_vec[3] > 0.5) else 0.0

    engagement = (
        0.3 * (lv + rv) + 0.3 * bm +
        0.2 * emotion_vec[0] + 0.2 * emotion_vec[1]
    )
    if low_motion_flag:
        engagement *= 0.6
    if non_expressive:
        engagement *= 0.7
    engagement = max(0.0, min(engagement, 1.0))

    feature = np.concatenate([
        emotion_vec,
        np.array([lv, rv, sh, ht, bm, face_conf,
                  low_motion_flag, non_expressive, engagement], dtype=np.float32),
    ])
    return feature, new_lw, new_rw


class EnthusiasmPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.available = False

        if not os.path.isfile(model_path):
            print(f"  [WARNING] Enthusiasm model not found: {model_path}")
            return
        if not os.path.isfile(scaler_path):
            print(f"  [WARNING] Enthusiasm scaler not found: {scaler_path}")
            return
        if not JOBLIB_AVAILABLE:
            print("  [WARNING] joblib not installed — enthusiasm detection disabled.")
            return

        checkpoint = torch.load(model_path, map_location="cpu")
        cfg = checkpoint.get("cfg", ENTHU_CFG)
        self.model = EnthusiasmLSTM(
            n_features=cfg.get("n_features", 15),
            hidden=cfg.get("hidden_size", 128),
            layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.4),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.scaler     = joblib.load(scaler_path)
        self.seq_len    = cfg.get("seq_len", 16)
        self.stride     = cfg.get("stride", 4)
        self.threshold  = cfg.get("threshold", 0.45)
        self.available  = True
        self._pose_model = None

        # Running buffers
        self._raw_features  = []
        self._norm_buffer   = []
        self._prev_lw       = None
        self._prev_rw       = None
        self._frame_probs   = []
        self._smooth_prob   = 0.5

        if MEDIAPIPE_AVAILABLE:
            self._pose_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )

    def update(self, frame_bgr) -> Tuple[float, str]:
        """Process one frame; return (probability, label_str)."""
        if not self.available:
            return 0.5, "N/A"

        feat, self._prev_lw, self._prev_rw = build_frame_features_fn(
            frame_bgr, self._pose_model, self._prev_lw, self._prev_rw
        )
        self._raw_features.append(feat)

        # Normalise
        norm = self.scaler.transform(feat.reshape(1, -1)).flatten().astype(np.float32)
        self._norm_buffer.append(norm)

        prob = self._smooth_prob
        if len(self._norm_buffer) >= self.seq_len:
            window = np.array(self._norm_buffer[-self.seq_len:], dtype=np.float32)
            xb     = torch.tensor(window).unsqueeze(0)
            with torch.no_grad():
                raw_prob = torch.softmax(self.model(xb), dim=1)[0, 1].item() * 0.8
            self._smooth_prob = 0.7 * raw_prob + 0.3 * self._smooth_prob
            prob = self._smooth_prob

        label = "ENTHU" if prob > self.threshold else "NOT ENTHU"
        return prob, label

    def final_summary(self) -> dict:
        if not self.available or len(self._raw_features) == 0:
            return {}

        raw_arr   = np.array(self._raw_features, dtype=np.float32)
        norm_arr  = self.scaler.transform(raw_arr).astype(np.float32)
        T         = len(norm_arr)
        frame_probs = np.full(T, np.nan)

        for start in range(0, T - self.seq_len + 1, self.stride):
            window = norm_arr[start: start + self.seq_len]
            xb     = torch.tensor(window).unsqueeze(0)
            with torch.no_grad():
                prob = torch.softmax(self.model(xb), dim=1)[0, 1].item() * 0.8
            frame_probs[start:start + self.seq_len] = np.where(
                np.isnan(frame_probs[start:start + self.seq_len]),
                prob,
                (frame_probs[start:start + self.seq_len] + prob) / 2,
            )

        for i in range(T):
            if np.isnan(frame_probs[i]):
                frame_probs[i] = frame_probs[i - 1] if i > 0 else 0.5
        for i in range(1, T):
            frame_probs[i] = 0.7 * frame_probs[i] + 0.3 * frame_probs[i - 1]

        labels       = (frame_probs > self.threshold).astype(int)
        enthu_pct    = float(labels.mean()) * 100
        not_enthu_pct = 100 - enthu_pct

        engagement_col = raw_arr[:, 14]
        motion_col     = raw_arr[:, 10]

        if enthu_pct > 70:
            verdict = "HIGHLY ENTHUSIASTIC 🔥"
        elif not_enthu_pct > 70:
            verdict = "NOT ENTHUSIASTIC ⚠️"
        else:
            dom = "Enthusiastic" if enthu_pct >= not_enthu_pct else "Not Enthusiastic"
            dom_pct = enthu_pct if enthu_pct >= not_enthu_pct else not_enthu_pct
            verdict = f"MIXED — Dominant: {dom} ({dom_pct:.1f}%) ⚖️"

        return {
            "enthu_pct":      round(enthu_pct, 1),
            "not_enthu_pct":  round(not_enthu_pct, 1),
            "verdict":        verdict,
            "avg_engagement": round(float(np.mean(engagement_col)), 4),
            "max_engagement": round(float(np.max(engagement_col)), 4),
            "avg_motion":     round(float(np.mean(motion_col)), 4),
            "frame_probs":    frame_probs.tolist(),
        }

    def close(self):
        if self._pose_model is not None:
            self._pose_model.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PART C : DISPLAY / OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════

MODE_COLORS = {
    "boardonly":   (40, 210, 60),
    "pptonly":     (40, 130, 255),
    "boardandppt": (0, 160, 255),
}
MODE_LABELS = {
    "boardonly":   "BLACKBOARD ONLY",
    "pptonly":     "PPT / SCREEN ONLY",
    "boardandppt": "BOARD + PPT",
}
ENTHU_COLORS = {
    "ENTHU":     (0, 220, 100),
    "NOT ENTHU": (0, 100, 220),
    "N/A":       (120, 120, 120),
}


def draw_overlay(frame, mode_smooth, mode_raw, mode_conf, mode_probs,
                 class_names, enthu_label, enthu_prob, fps, ts):
    h, w = frame.shape[:2]

    # ── Top banner ────────────────────────────────────────────────────────────
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, 140), (10, 10, 14), -1)
    cv2.addWeighted(banner, 0.65, frame, 0.35, 0, frame)

    # Teaching mode label
    mode_col = MODE_COLORS.get(mode_smooth, (200, 200, 200))
    cv2.putText(frame, f"MODE: {MODE_LABELS.get(mode_smooth, mode_smooth)}",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.90, mode_col, 2, cv2.LINE_AA)

    # Raw frame + conf
    raw_col = MODE_COLORS.get(mode_raw, (160, 160, 160))
    cv2.putText(frame, f"frame:{mode_raw}  {mode_conf*100:.0f}%",
                (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, raw_col, 1, cv2.LINE_AA)

    # Enthusiasm
    ecol = ENTHU_COLORS.get(enthu_label, (180, 180, 180))
    cv2.putText(frame, f"ENTHUSIASM: {enthu_label}  {enthu_prob*100:.0f}%",
                (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.72, ecol, 2, cv2.LINE_AA)

    # FPS + time top-right
    m, s = divmod(int(ts), 60)
    cv2.putText(frame, f"{m:02d}:{s:02d}  {fps:.0f}fps",
                (w - 165, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1)

    # Class probability bars
    by = 100
    for cn, p in zip(class_names, mode_probs):
        bw = int(200 * p)
        c  = MODE_COLORS.get(cn, (100, 100, 100))
        cv2.rectangle(frame, (12, by), (12 + bw, by + 13), c, -1)
        cv2.putText(frame, f"{cn}: {p*100:.0f}%",
                    (222, by + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
        by += 17

    # ── Bottom status bar ────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 44), (w, h), (0, 0, 0), -1)

    # Left: teaching mode
    cv2.putText(frame, MODE_LABELS.get(mode_smooth, mode_smooth),
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, mode_col, 2, cv2.LINE_AA)

    # Right: enthusiasm
    label_text = f"{enthu_label}  {enthu_prob*100:.0f}%"
    tw, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)[0], 0
    cv2.putText(frame, label_text,
                (w - tw[0] - 14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, ecol, 2, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  PART D : SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

MODE_DESCRIPTIONS = {
    "boardonly":   "Teacher used only the blackboard — writing and explaining with chalk.",
    "pptonly":     "Teacher used only PPT/projector — presenting digital slides.",
    "boardandppt": "Teacher combined blackboard and PPT — dual-mode instruction.",
}

def fmt_time(sec):
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def print_summary(video_path, all_mode_preds, class_names,
                  enthu_summary, output_path, fps_src=25.0, true_duration=0.0):
    total   = len(all_mode_preds)
    if total == 0:
        return

    labels  = [p[0] for p in all_mode_preds]
    counts  = Counter(labels)
    frame_dur = 1.0 / max(fps_src, 1.0)

    time_per_mode = {cls: 0.0 for cls in class_names}
    for i, (lbl, ts) in enumerate(all_mode_preds):
        gap = (all_mode_preds[i + 1][1] - ts) if i + 1 < len(all_mode_preds) else frame_dur
        gap = gap if gap > 0 else frame_dur
        time_per_mode[lbl] = time_per_mode.get(lbl, 0.0) + gap

    total_time = sum(time_per_mode.values()) or (total * frame_dur)
    if total_time <= 0:
        for cls in class_names:
            time_per_mode[cls] = counts.get(cls, 0) * frame_dur
        total_time = total * frame_dur

    pct_board   = time_per_mode.get("boardonly", 0.0) / total_time * 100
    pct_ppt     = time_per_mode.get("pptonly",   0.0) / total_time * 100
    mode_verdict = "boardandppt" if (pct_board > 5 and pct_ppt > 5) else counts.most_common(1)[0][0]

    sep  = "═" * 64
    thin = "─" * 64

    print(f"\n{sep}")
    print("  TEACHER DASHBOARD  —  COMBINED ANALYSIS REPORT")
    print(sep)
    print(f"\n  Video            : {Path(video_path).name}")
    vid_dur = true_duration if true_duration > 0 else total_time
    print(f"  Duration         : {fmt_time(vid_dur)}")

    # ── Teaching Mode ─────────────────────────────────────────────────────────
    print(f"\n  {thin}")
    print("  📋  TEACHING MODE BREAKDOWN")
    print(f"  {thin}")
    for cls in class_names:
        t_cls  = time_per_mode.get(cls, 0.0)
        pct_t  = t_cls / total_time * 100 if total_time > 0 else 0.0
        bar    = "█" * int(pct_t / 5)
        label  = MODE_LABELS.get(cls, cls)
        if t_cls > 0:
            print(f"  {label:<22}  {fmt_time(t_cls)}  ({pct_t:5.1f}%)  {bar}")
        else:
            print(f"  {label:<22}  {fmt_time(t_cls)}  ({pct_t:5.1f}%)")

    print(f"\n  MODE VERDICT     : {MODE_LABELS.get(mode_verdict, mode_verdict)}")
    print(f"  CONCLUSION       : {MODE_DESCRIPTIONS.get(mode_verdict, mode_verdict)}")

    # ── Enthusiasm ────────────────────────────────────────────────────────────
    print(f"\n  {thin}")
    print("  🔥  ENTHUSIASM ANALYSIS")
    print(f"  {thin}")

    if enthu_summary:
        ep   = enthu_summary.get("enthu_pct", 0)
        nep  = enthu_summary.get("not_enthu_pct", 0)
        e_bar = "█" * int(ep / 5)
        n_bar = "█" * int(nep / 5)
        print(f"  Enthusiastic     : [{e_bar:<20}]  {ep:5.1f}%")
        print(f"  Not Enthusiastic : [{n_bar:<20}]  {nep:5.1f}%")
        print(f"\n  Avg Engagement   : {enthu_summary.get('avg_engagement', 0):.4f}")
        print(f"  Peak Engagement  : {enthu_summary.get('max_engagement', 0):.4f}")
        print(f"  Avg Body Motion  : {enthu_summary.get('avg_motion', 0):.4f}")
        print(f"\n  ENTHU VERDICT    : {enthu_summary.get('verdict', 'N/A')}")
    else:
        print("  Enthusiasm model not available — skipped.")

    # ── Combined Insight ──────────────────────────────────────────────────────
    print(f"\n  {thin}")
    print("  🎓  OVERALL TEACHER INSIGHT")
    print(f"  {thin}")
    if enthu_summary:
        ep = enthu_summary.get("enthu_pct", 0)
        mode_str = MODE_LABELS.get(mode_verdict, mode_verdict)
        if ep > 70:
            energy = "high energy and enthusiasm"
        elif ep > 40:
            energy = "moderate enthusiasm"
        else:
            energy = "low energy (room for improvement)"
        print(f"  This teacher primarily uses '{mode_str}' and shows {energy}.")
        if ep > 60 and mode_verdict == "boardandppt":
            print("  ✔ Excellent engagement — combining both tools with active presence!")
        elif ep < 40:
            print("  ✎ Consider adding more interactive or expressive teaching elements.")
    else:
        print(f"  Teaching mode: {MODE_LABELS.get(mode_verdict, mode_verdict)}")

    print(f"\n  {thin}")
    print(f"  Annotated video  : {output_path}")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART E : MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def find_file(folder: Path, names: List[str]) -> Optional[str]:
    """Search for a file by any of the given names in folder and sub-folders."""
    for name in names:
        for p in folder.rglob(name):
            if p.is_file():
                return str(p)
    return None


def pick_random_video(folder: Path) -> Optional[str]:
    videos = [f for ext in VIDEO_EXTS
               for f in list(folder.rglob(f"*{ext}")) +
                        list(folder.rglob(f"*{ext.upper()}"))]
    if not videos:
        return None
    return str(random.choice(videos))


def main():
    ap = argparse.ArgumentParser(description="Combined Teacher Analyzer")
    ap.add_argument("--folder", default=DEFAULT_TEACHER_FOLDER,
                    help="Path to the 'teacher' folder containing videos + models")
    ap.add_argument("--skip", type=int, default=2,
                    help="Analyse 1-in-N frames for teaching mode (default 2)")
    ap.add_argument("--no_display", action="store_true",
                    help="Don't show live window (useful for headless servers)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"\nERROR: Folder not found: {folder}")
        print("Create a 'teacher' folder and put your videos + model files inside.")
        return

    print("\n" + "═" * 60)
    print("  Teacher Analyzer  —  Combined Analysis System")
    print("═" * 60)

    # ── Pick video ────────────────────────────────────────────────────────────
    video_path = pick_random_video(folder)
    if not video_path:
        print(f"\nERROR: No video files found in: {folder}")
        return
    print(f"\n  📹  Selected video : {Path(video_path).name}")

    # ── Find models ───────────────────────────────────────────────────────────
    teaching_model_path = find_file(folder, ["best_model.pth"])
    enthu_model_path    = find_file(folder, ["enthusiasm_lstm.pt"])
    enthu_scaler_path   = find_file(folder, ["feature_scaler.pkl"])

    if not teaching_model_path:
        print(f"\nERROR: Teaching mode model (best_model.pth) not found in {folder}")
        print("Please copy best_model.pth into the teacher folder.")
        return

    print(f"  🤖  Teaching model    : {Path(teaching_model_path).name}")
    if enthu_model_path:
        print(f"  🤖  Enthusiasm model  : {Path(enthu_model_path).name}")
    else:
        print("  ⚠️   Enthusiasm model not found — enthusiasm detection disabled.")
    print(f"  💻  Device            : {DEVICE}")

    # ── Output path ───────────────────────────────────────────────────────────
    outputs_dir = folder / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    vid_stem   = Path(video_path).stem
    run_ts     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_video  = str(outputs_dir / f"combined_output_{vid_stem}_{run_ts}.mp4")
    out_json   = str(outputs_dir / f"combined_report_{vid_stem}_{run_ts}.json")
    print(f"  📂  Output folder     : {outputs_dir}\n")

    # ── Load predictors ───────────────────────────────────────────────────────
    teaching_pred = TeachingPredictor(teaching_model_path)
    print(f"  Teaching classes : {teaching_pred.class_names}")

    if enthu_model_path and enthu_scaler_path:
        enthu_pred = EnthusiasmPredictor(enthu_model_path, enthu_scaler_path)
        if enthu_pred.available:
            print("  Enthusiasm predictor : ready")
        else:
            print("  Enthusiasm predictor : failed to load")
    else:
        enthu_pred = EnthusiasmPredictor("__missing__", "__missing__")

    print("\n  Q=quit   S=screenshot   P=pause\n")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    fps_src          = cap.get(cv2.CAP_PROP_FPS) or 25
    width            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    true_duration    = total_frames_cap / max(fps_src, 1.0)
    delay            = max(1, int(1000 / fps_src))

    print(f"  Video : {width}x{height}  {fps_src:.1f}fps  "
          f"{total_frames_cap} frames  {fmt_time(true_duration)}\n")

    writer = cv2.VideoWriter(
        out_video, cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src, (width, height)
    )

    frame_n = 0;  t_prev = time.time();  fps_disp = 0.0
    paused  = False;  shot_n = 0;  display = None

    last_mode_raw = last_mode_smooth = "..."
    last_mode_conf  = 0.0
    last_mode_probs = np.zeros(len(teaching_pred.class_names))
    last_enthu_label = "N/A";  last_enthu_prob = 0.5

    all_mode_preds = []     # (smooth_label, timestamp)

    # Enthusiasm is heavy — run it every ENTHU_SKIP frames
    ENTHU_SKIP = max(1, int(fps_src / ENTHU_CFG["target_fps"]))

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  Video finished.")
                break

            frame_n += 1
            now      = time.time()
            fps_disp = 0.88 * fps_disp + 0.12 / max(now - t_prev, 1e-6)
            t_prev   = now
            ts       = frame_n / max(fps_src, 1.0)

            # ── Teaching mode prediction ──────────────────────────────────────
            if frame_n % args.skip == 0:
                last_mode_raw, last_mode_smooth, last_mode_conf, last_mode_probs = \
                    teaching_pred.predict(frame)

            all_mode_preds.append((last_mode_smooth, ts))

            # ── Enthusiasm prediction ─────────────────────────────────────────
            if frame_n % ENTHU_SKIP == 0 and enthu_pred.available:
                last_enthu_prob, last_enthu_label = enthu_pred.update(frame)

            # ── Draw ──────────────────────────────────────────────────────────
            display = draw_overlay(
                frame.copy(),
                last_mode_smooth, last_mode_raw, last_mode_conf, last_mode_probs,
                teaching_pred.class_names,
                last_enthu_label, last_enthu_prob,
                fps_disp, ts,
            )
            writer.write(display)

        if display is not None and not args.no_display:
            cv2.imshow("Teacher Analyzer — Combined Dashboard", display)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = f"screenshot_{shot_n:04d}.jpg"
            cv2.imwrite(fname, display)
            print(f"  Saved: {fname}")
            shot_n += 1
        elif key == ord("p"):
            paused = not paused
            print("  Paused." if paused else "  Resumed.")

    cap.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ── Final summaries ───────────────────────────────────────────────────────
    enthu_summary = enthu_pred.final_summary() if enthu_pred.available else {}
    if enthu_pred.available:
        enthu_pred.close()

    print_summary(video_path, all_mode_preds, teaching_pred.class_names,
                  enthu_summary, out_video, fps_src, true_duration)

    # ── Save JSON report ──────────────────────────────────────────────────────
    mode_counts = Counter([p[0] for p in all_mode_preds])
    total_time  = true_duration if true_duration > 0 else (frame_n / max(fps_src, 1.0))
    report = {
        "video":          Path(video_path).name,
        "duration_sec":   round(total_time, 2),
        "device":         str(DEVICE),
        "teaching_mode": {
            "frame_counts": dict(mode_counts),
            "verdict":      Counter([p[0] for p in all_mode_preds]).most_common(1)[0][0]
                            if all_mode_preds else "unknown",
        },
        "enthusiasm": enthu_summary if enthu_summary else {"status": "model not loaded"},
        "output_video": out_video,
    }

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  📄  JSON report saved : {out_json}\n")


if __name__ == "__main__":
    main()
