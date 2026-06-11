import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pyautogui
import os
import urllib.request
import math

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

pyautogui.FAILSAFE = False
screen_width, screen_height = pyautogui.size()

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam not detected.")
    exit()

# ── CALIBRATION STATE ─────────────────────────────────────────
CAL_DURATION = 1200
CAL_FILE = "calibration.txt"

SMOOTHING = 0.4
prev_x, prev_y = screen_width // 2, screen_height // 2

# ── CLICK STATE ───────────────────────────────────────────────
PINCH_THRESHOLD = 40
CLICK_COOLDOWN = 20
click_cooldown_counter = 0

# Try to load saved calibration
if os.path.exists(CAL_FILE):
    with open(CAL_FILE, "r") as f:
        vals = f.read().split()
        cal_min_x, cal_max_x = float(vals[0]), float(vals[1])
        cal_min_y, cal_max_y = float(vals[2]), float(vals[3])
    calibrating = False
    cal_frames = CAL_DURATION
    print(f"Loaded saved calibration: x={cal_min_x:.2f}-{cal_max_x:.2f}, y={cal_min_y:.2f}-{cal_max_y:.2f}")
    print("SPIDER CURSOR ONLINE — skipping calibration")
else:
    calibrating = True
    cal_min_x, cal_max_x = 1.0, 0.0
    cal_min_y, cal_max_y = 1.0, 0.0
    cal_frames = 0
    print("CALIBRATION MODE: Move your index finger to all corners of the frame")
    print("SPIDER CURSOR ONLINE")

frame_timestamp = 0

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp += 1
    results = hand_landmarker.detect_for_video(mp_image, frame_timestamp)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:

            # Draw all landmarks
            for lm in hand:
                cx = int(lm.x * frame_width)
                cy = int(lm.y * frame_height)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Index fingertip
            tip = hand[8]
            tx = int(tip.x * frame_width)
            ty = int(tip.y * frame_height)
            cv2.circle(frame, (tx, ty), 12, (0, 0, 255), -1)

            # ── PINCH DETECTION ───────────────────────────────
            thumb_tip = hand[4]
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            cv2.circle(frame, (thumb_x, thumb_y), 12, (255, 0, 0), -1)

            pinch_dist = math.sqrt((tx - thumb_x)**2 + (ty - thumb_y)**2)

            cv2.line(frame, (tx, ty), (thumb_x, thumb_y), (255, 255, 0), 2)

            cv2.putText(frame, f"PINCH: {int(pinch_dist)}px",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if click_cooldown_counter == 0 and pinch_dist < PINCH_THRESHOLD:
                pyautogui.click()
                click_cooldown_counter = CLICK_COOLDOWN
                cv2.putText(frame, "CLICK!", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                print("CLICK fired")

            if click_cooldown_counter > 0:
                click_cooldown_counter -= 1

            if calibrating:
                cal_min_x = min(cal_min_x, tip.x)
                cal_max_x = max(cal_max_x, tip.x)
                cal_min_y = min(cal_min_y, tip.y)
                cal_max_y = max(cal_max_y, tip.y)
                cal_frames += 1

                bx1 = int(cal_min_x * frame_width)
                by1 = int(cal_min_y * frame_height)
                bx2 = int(cal_max_x * frame_width)
                by2 = int(cal_max_y * frame_height)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

                progress = int((cal_frames / CAL_DURATION) * frame_width)
                cv2.rectangle(frame, (0, frame_height - 20), (progress, frame_height), (0, 255, 255), -1)
                cv2.putText(frame, "CALIBRATING - sweep finger to all corners",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if cal_frames >= CAL_DURATION:
                    calibrating = False
                    pad_x = (cal_max_x - cal_min_x) * 0.05
                    pad_y = (cal_max_y - cal_min_y) * 0.05
                    cal_min_x = max(0, cal_min_x - pad_x)
                    cal_max_x = min(1, cal_max_x + pad_x)
                    cal_min_y = max(0, cal_min_y - pad_y)
                    cal_max_y = min(1, cal_max_y + pad_y)
                    print(f"Calibration done! Zone: x={cal_min_x:.2f}-{cal_max_x:.2f}, y={cal_min_y:.2f}-{cal_max_y:.2f}")
                    with open(CAL_FILE, "w") as f:
                        f.write(f"{cal_min_x} {cal_max_x} {cal_min_y} {cal_max_y}")
                    print("Calibration saved!")

            else:
                range_x = cal_max_x - cal_min_x
                range_y = cal_max_y - cal_min_y

                mapped_x = (tip.x - cal_min_x) / range_x if range_x > 0 else 0.5
                mapped_y = (tip.y - cal_min_y) / range_y if range_y > 0 else 0.5

                mapped_x = max(0.0, min(1.0, mapped_x))
                mapped_y = max(0.0, min(1.0, mapped_y))

                target_x = int(mapped_x * screen_width)
                target_y = int(mapped_y * screen_height)

                target_x = max(0, min(screen_width - 1, target_x))
                target_y = max(0, min(screen_height - 1, target_y))

                smooth_x = int(prev_x * SMOOTHING + target_x * (1 - SMOOTHING))
                smooth_y = int(prev_y * SMOOTHING + target_y * (1 - SMOOTHING))

                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                prev_x, prev_y = smooth_x, smooth_y

                cv2.putText(frame, "SPIDER CURSOR ACTIVE",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("SPIDERSENSE - PETER PARKER PROTOCOL", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()