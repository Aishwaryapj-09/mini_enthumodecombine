keep the pth and related files in datsets where video thr that folder 


# Teacher Analyzer — Combined Analysis System
## What it does
Picks **one random video** from your `teacher/` folder and runs **two analyses simultaneously**:

| Analysis | What it detects |
|---|---|
| 🖥️ **Teaching Mode** | `BLACKBOARD ONLY` / `PPT ONLY` / `BOARD + PPT` |
| 🔥 **Enthusiasm** | `ENTHUSIASTIC` / `NOT ENTHUSIASTIC` + percentage |

Both results appear live on the annotated video **and** in a final dashboard summary.

---

## Folder Setup (REQUIRED)

```
teacher/
  your_video1.mp4          ← any classroom videos (mixed is fine)
  your_video2.mp4
  best_model.pth           ← teaching mode model  (from model_trainer.py)
  enthusiasm_lstm.pt       ← enthusiasm model     (from enthusiasm training)
  feature_scaler.pkl       ← scaler               (from enthusiasm training)
```

> **All three `.pth`/`.pt`/`.pkl` files must be inside the `teacher/` folder.**
> Sub-folders are fine — the script searches recursively.

---

## Install requirements

```bash
pip install torch torchvision opencv-python mediapipe scikit-learn joblib
```

> `mediapipe` is optional — if missing, enthusiasm still runs in motion-only mode.

---

## Run

```bash
# Default — looks for ./teacher folder
python teacher_analyzer.py

# Custom folder
python teacher_analyzer.py --folder "D:/myproject/teacher"

# Headless (no display window, just save video)
python teacher_analyzer.py --no_display

# Faster (analyse every frame for teaching mode)
python teacher_analyzer.py --skip 1
```

---

## Output

All outputs go to `teacher/outputs/`:

```
teacher/outputs/
  combined_output_<video>_<timestamp>.mp4    ← annotated video
  combined_report_<video>_<timestamp>.json   ← full JSON report
```

---

## Live Video Overlay

| Area | Shows |
|---|---|
| **Top left** | Teaching mode (colour-coded) |
| **Middle left** | Enthusiasm label + confidence % |
| **Probability bars** | Per-class breakdown |
| **Bottom bar** | Mode (left) and Enthusiasm (right) |

### Keyboard controls
| Key | Action |
|---|---|
| `Q` or `ESC` | Quit |
| `S` | Save screenshot |
| `P` | Pause / Resume |

---

## Final Dashboard (printed in terminal)

```
════════════════════════════════════════════════════════════════
  TEACHER DASHBOARD  —  COMBINED ANALYSIS REPORT
════════════════════════════════════════════════════════════════

  📋  TEACHING MODE BREAKDOWN
  ─────────────────────────────────────────────────────────────────
  BLACKBOARD ONLY        00:45  ( 37.5%)  ████████
  PPT / SCREEN ONLY      00:30  ( 25.0%)  █████
  BOARD + PPT            00:45  ( 37.5%)  ████████

  MODE VERDICT     : BOARD + PPT
  CONCLUSION       : Teacher combined blackboard and PPT

  🔥  ENTHUSIASM ANALYSIS
  ─────────────────────────────────────────────────────────────────
  Enthusiastic     : [████████████        ]   62.3%
  Not Enthusiastic : [████████            ]   37.7%

  Avg Engagement   : 0.4821
  Peak Engagement  : 0.8903
  Avg Body Motion  : 0.2341

  ENTHU VERDICT    : MIXED — Dominant: Enthusiastic (62.3%) ⚖️

  🎓  OVERALL TEACHER INSIGHT
  ─────────────────────────────────────────────────────────────────
  This teacher primarily uses 'BOARD + PPT' and shows moderate enthusiasm.
```

---

## No retraining needed
This script **only loads** the pre-trained models. It never calls any training code.
