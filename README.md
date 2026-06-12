# SpiderSense 🕷️

A Spider-Man themed, gesture-controlled computer interface built with Python, OpenCV, and MediaPipe. Control your mouse, click, scroll, and launch apps — all with your hand, tracked live through your webcam.

> *"With great power comes great... cursor control."*

---

## Features

- **Real-time hand tracking** using MediaPipe's HandLandmarker (21 landmarks)
- **Cursor control** — move your mouse by moving your index finger
- **Self-calibrating tracking zone** — sweep your finger to all corners once, and SpiderSense adapts to your range of motion (saved to `calibration.txt` for future runs)
- **Pinch-to-click** — bring your thumb and index finger together to click
- **Two-finger scroll** — extend index + middle finger and move up/down to scroll
- **Thumbs-up app launcher** — hold a thumbs-up gesture to launch Safari
- **Spider-Man HUD** — targeting reticle, web logo, and live status messages (`TARGET LOCKED`, `WEB-SHOT FIRED!`, `THREAT ANALYSIS ACTIVE`)

---

## How It Works

```
Webcam → OpenCV (capture frame)
       → MediaPipe (detect 21 hand landmarks)
       → Gesture logic (cursor / pinch / scroll / thumbs-up)
       → PyAutoGUI (move mouse, click, scroll, launch apps)
       → OpenCV (draw HUD overlay)
       → Display
```

Each frame is processed in this loop, ~30 times per second, creating real-time responsiveness.

### Gestures

| Gesture | Action |
|---|---|
| Index finger extended, move hand | Move cursor |
| Thumb + index finger pinch | Click |
| Index + middle finger extended, move up/down | Scroll |
| Thumbs-up, hold ~1 second | Launch Safari |

---

## Setup

### 1. Clone and enter the project
```bash
git clone https://github.com/chirrupings123/SpiderSense
cd spidersense
```

### 2. Create and activate a virtual environment
```bash
python -m venv spidersense_env
source spidersense_env/bin/activate   # macOS/Linux
spidersense_env\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install opencv-python mediapipe pyautogui numpy
```

> **macOS note:** if you hit an `SSL: CERTIFICATE_VERIFY_FAILED` error on first run (when the hand model auto-downloads), run:
> ```bash
> open "/Applications/Python 3.x/Install Certificates.command"
> ```

---

## Usage

```bash
python spidersense.py
```
or

```bash
python3 spidersense.py
```

- **First run:** SpiderSense enters calibration mode. Sweep your index finger to all four corners of the camera frame until the progress bar fills. This is saved automatically.
- **Move cursor:** point with your index finger.
- **Click:** pinch thumb and index finger together.
- **Scroll:** extend index + middle finger and move your hand up or down.
- **Launch Safari:** make a thumbs-up and hold it for about a second.
- **Quit:** press `q`.

To recalibrate, delete `calibration.txt` and run again.

---

## Tuning

A few constants near the top of `spidersense.py` control feel and sensitivity:

| Constant | Purpose |
|---|---|
| `SMOOTHING` | Cursor smoothing factor (higher = smoother but more lag) |
| `PINCH_THRESHOLD` | Pixel distance for pinch-click to register |
| `CLICK_COOLDOWN` | Frames between allowed clicks |
| `SCROLL_SENSITIVITY` | How far scroll moves per unit of hand motion |
| `SCROLL_DEADZONE` | Minimum hand movement before scroll triggers (filters jitter) |
| `THUMBS_UP_HOLD_FRAMES` | How long thumbs-up must be held to launch an app |
| `APP_LAUNCH_COOLDOWN` | Frames before another app launch can trigger |

---

## Tech Stack

- **Python**
- **OpenCV** — webcam capture and HUD rendering
- **MediaPipe** (Tasks API) — hand landmark detection
- **PyAutoGUI** — mouse and scroll control
- **NumPy**

---

## Possible Extensions

- Additional gestures (e.g. fist-to-drag, right-click)
- Configurable app-launch gestures
- On-screen settings menu for live tuning
- Support for multiple monitors
