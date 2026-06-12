import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pyautogui
import os
import urllib.request
import math
import subprocess


# ── CONFIG ─────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CAL_FILE = "calibration.txt"
CAL_DURATION = 1200

SMOOTHING = 0.4
PINCH_THRESHOLD = 60
CLICK_COOLDOWN = 20

# Gesture settings
THUMBS_UP_HOLD_FRAMES = 30      # ~1 second at 30fps
APP_LAUNCH_COOLDOWN = 60        # frames to wait after launching before re-triggering
SCROLL_SENSITIVITY = 800         # multiplier for scroll delta
SCROLL_DEADZONE = 0.008          # ignore tiny jitter movements


# ── SETUP HELPERS ────────────────────────────────────────────────

def ensure_model_downloaded():
    """Downloads the MediaPipe hand model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading SpiderSense hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def create_hand_landmarker():
    """Creates and returns a configured MediaPipe HandLandmarker."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options)


# ── CALIBRATION ────────────────────────────────────────────────────

def load_calibration():
    """Loads saved calibration zone from file. Returns None if not found."""
    if not os.path.exists(CAL_FILE):
        return None
    with open(CAL_FILE, "r") as f:
        vals = f.read().split()
    return {
        "min_x": float(vals[0]), "max_x": float(vals[1]),
        "min_y": float(vals[2]), "max_y": float(vals[3])
    }


def save_calibration(cal):
    """Saves calibration zone to file."""
    with open(CAL_FILE, "w") as f:
        f.write(f"{cal['min_x']} {cal['max_x']} {cal['min_y']} {cal['max_y']}")
    print("Calibration saved!")


def run_calibration_step(frame, tip, cal, cal_frames, frame_width, frame_height):
    """
    Updates calibration bounds for one frame.
    Returns (updated_cal, updated_cal_frames, is_done).
    """
    cal["min_x"] = min(cal["min_x"], tip.x)
    cal["max_x"] = max(cal["max_x"], tip.x)
    cal["min_y"] = min(cal["min_y"], tip.y)
    cal["max_y"] = max(cal["max_y"], tip.y)
    cal_frames += 1

    # Draw calibration zone box
    bx1 = int(cal["min_x"] * frame_width)
    by1 = int(cal["min_y"] * frame_height)
    bx2 = int(cal["max_x"] * frame_width)
    by2 = int(cal["max_y"] * frame_height)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

    # Draw progress bar
    progress = int((cal_frames / CAL_DURATION) * frame_width)
    cv2.rectangle(frame, (0, frame_height - 20), (progress, frame_height), (0, 255, 255), -1)

    draw_hud_text(frame, [
        "PETER PARKER PROTOCOL INITIATED",
        "CALIBRATING SPIDER-SENSE...",
        "Sweep index finger to all corners"
    ], y_start=30, color=(0, 255, 255))

    is_done = cal_frames >= CAL_DURATION
    if is_done:
        pad_x = (cal["max_x"] - cal["min_x"]) * 0.05
        pad_y = (cal["max_y"] - cal["min_y"]) * 0.05
        cal["min_x"] = max(0, cal["min_x"] - pad_x)
        cal["max_x"] = min(1, cal["max_x"] + pad_x)
        cal["min_y"] = max(0, cal["min_y"] - pad_y)
        cal["max_y"] = min(1, cal["max_y"] + pad_y)
        print(f"Calibration done! Zone: x={cal['min_x']:.2f}-{cal['max_x']:.2f}, "
              f"y={cal['min_y']:.2f}-{cal['max_y']:.2f}")
        save_calibration(cal)

    return cal, cal_frames, is_done


# ── CURSOR CONTROL ────────────────────────────────────────────────

def map_to_screen(tip, cal, screen_width, screen_height):
    """Maps a normalised fingertip position to screen pixel coordinates."""
    range_x = cal["max_x"] - cal["min_x"]
    range_y = cal["max_y"] - cal["min_y"]

    mapped_x = (tip.x - cal["min_x"]) / range_x if range_x > 0 else 0.5
    mapped_y = (tip.y - cal["min_y"]) / range_y if range_y > 0 else 0.5

    mapped_x = max(0.0, min(1.0, mapped_x))
    mapped_y = max(0.0, min(1.0, mapped_y))

    target_x = int(mapped_x * screen_width)
    target_y = int(mapped_y * screen_height)

    target_x = max(0, min(screen_width - 1, target_x))
    target_y = max(0, min(screen_height - 1, target_y))

    return target_x, target_y


def smooth_position(prev_x, prev_y, target_x, target_y, smoothing):
    """Blends previous position with target position for smooth movement."""
    smooth_x = int(prev_x * smoothing + target_x * (1 - smoothing))
    smooth_y = int(prev_y * smoothing + target_y * (1 - smoothing))
    return smooth_x, smooth_y


# ── PINCH / CLICK ─────────────────────────────────────────────────

def get_pinch_distance(tx, ty, thumb_x, thumb_y):
    """Returns Euclidean distance between index tip and thumb tip."""
    return math.sqrt((tx - thumb_x) ** 2 + (ty - thumb_y) ** 2)


def handle_pinch_click(frame, pinch_dist, cooldown_counter):
    """
    Fires a click if pinch is detected and cooldown is ready.
    Returns updated cooldown_counter.
    """
    if cooldown_counter == 0 and pinch_dist < PINCH_THRESHOLD:
        pyautogui.click()
        cooldown_counter = CLICK_COOLDOWN
        cv2.putText(frame, "CLICK!", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        print("CLICK fired")

    if cooldown_counter > 0:
        cooldown_counter -= 1

    return cooldown_counter


# ── GESTURE DETECTION ─────────────────────────────────────────

def is_finger_extended(hand, tip_idx, pip_idx):
    """Returns True if a finger is extended (tip above its PIP knuckle)."""
    return hand[tip_idx].y < hand[pip_idx].y


def is_thumbs_up(hand):
    """
    Detects a thumbs-up gesture:
    - Thumb tip is well above the wrist
    - Index, middle, ring, pinky are all curled
    """
    thumb_up = hand[4].y < hand[0].y - 0.1  # thumb tip clearly above wrist
    index_curled = not is_finger_extended(hand, 8, 6)
    middle_curled = not is_finger_extended(hand, 12, 10)
    ring_curled = not is_finger_extended(hand, 16, 14)
    pinky_curled = not is_finger_extended(hand, 20, 18)

    return thumb_up and index_curled and middle_curled and ring_curled and pinky_curled


def is_scroll_gesture(hand):
    """
    Detects two-finger scroll gesture:
    - Index and middle fingers extended
    - Ring and pinky curled
    """
    index_ext = is_finger_extended(hand, 8, 6)
    middle_ext = is_finger_extended(hand, 12, 10)
    ring_curled = not is_finger_extended(hand, 16, 14)
    pinky_curled = not is_finger_extended(hand, 20, 18)

    return index_ext and middle_ext and ring_curled and pinky_curled


def open_app(app_name):
    """Launches a macOS application by name."""
    try:
        subprocess.Popen(["open", "-a", app_name])
        print(f"WEB-SHOOTER LAUNCH: {app_name}")
    except Exception as e:
        print(f"Failed to launch {app_name}: {e}")


# ── HUD ────────────────────────────────────────────────────────────

def draw_spider_logo(frame, x, y, size=30):
    """Draws a simple spider-web style logo using OpenCV shapes."""
    color = (255, 255, 255)
    cv2.circle(frame, (x, y), size, color, 2)
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        end_x = int(x + size * math.cos(rad))
        end_y = int(y + size * math.sin(rad))
        cv2.line(frame, (x, y), (end_x, end_y), color, 1)
    cv2.circle(frame, (x, y), int(size * 0.6), color, 1)
    cv2.circle(frame, (x, y), int(size * 0.3), color, 1)


def draw_hud_text(frame, lines, x=10, y_start=120, color=(0, 0, 255)):
    """Draws a list of status lines, stacked vertically."""
    for i, line in enumerate(lines):
        y = y_start + (i * 25)
        cv2.putText(frame, line, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_targeting_reticle(frame, cx, cy, size=25, color=(0, 0, 255)):
    """Draws a Spider-Man style targeting reticle (corner brackets)."""
    gap = size // 3
    cv2.line(frame, (cx - size, cy - size), (cx - size + gap, cy - size), color, 2)
    cv2.line(frame, (cx - size, cy - size), (cx - size, cy - size + gap), color, 2)
    cv2.line(frame, (cx + size, cy - size), (cx + size - gap, cy - size), color, 2)
    cv2.line(frame, (cx + size, cy - size), (cx + size, cy - size + gap), color, 2)
    cv2.line(frame, (cx - size, cy + size), (cx - size + gap, cy + size), color, 2)
    cv2.line(frame, (cx - size, cy + size), (cx - size, cy + size - gap), color, 2)
    cv2.line(frame, (cx + size, cy + size), (cx + size - gap, cy + size), color, 2)
    cv2.line(frame, (cx + size, cy + size), (cx + size, cy + size - gap), color, 2)


def get_hud_status_lines(pinch_dist):
    """Returns the dynamic status lines based on pinch distance."""
    lines = [
        "SPIDER-SENSE ONLINE",
        "WEBSHOOTER CALIBRATED",
        f"PINCH DIST: {int(pinch_dist)}px"
    ]
    if pinch_dist < PINCH_THRESHOLD:
        lines.append("WEB-SHOT FIRED!")
    elif pinch_dist < PINCH_THRESHOLD * 1.5:
        lines.append("TARGET LOCKED")
    else:
        lines.append("THREAT ANALYSIS ACTIVE")
    return lines


# ── MAIN ───────────────────────────────────────────────────────────

def main():
    ensure_model_downloaded()
    pyautogui.FAILSAFE = False
    screen_width, screen_height = pyautogui.size()

    hand_landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not detected.")
        exit()

    prev_x, prev_y = screen_width // 2, screen_height // 2
    click_cooldown_counter = 0

    # Gesture state
    thumbs_up_counter = 0
    app_launch_cooldown = 0
    prev_scroll_y = None

    cal = load_calibration()
    if cal:
        calibrating = False
        cal_frames = CAL_DURATION
        print(f"Loaded saved calibration: x={cal['min_x']:.2f}-{cal['max_x']:.2f}, "
              f"y={cal['min_y']:.2f}-{cal['max_y']:.2f}")
        print("SPIDER CURSOR ONLINE — skipping calibration")
    else:
        calibrating = True
        cal = {"min_x": 1.0, "max_x": 0.0, "min_y": 1.0, "max_y": 0.0}
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

        # Decrement app launch cooldown every frame regardless of hand presence
        if app_launch_cooldown > 0:
            app_launch_cooldown -= 1

        if results.hand_landmarks:
            for hand in results.hand_landmarks:

                # Draw all landmarks
                for lm in hand:
                    cx = int(lm.x * frame_width)
                    cy = int(lm.y * frame_height)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # Index fingertip + thumb tip
                tip = hand[8]
                tx, ty = int(tip.x * frame_width), int(tip.y * frame_height)
                cv2.circle(frame, (tx, ty), 8, (0, 0, 255), -1)
                draw_targeting_reticle(frame, tx, ty, size=25)

                thumb_tip = hand[4]
                thumb_x = int(thumb_tip.x * frame_width)
                thumb_y = int(thumb_tip.y * frame_height)
                cv2.circle(frame, (thumb_x, thumb_y), 12, (255, 0, 0), -1)
                cv2.line(frame, (tx, ty), (thumb_x, thumb_y), (255, 255, 0), 2)

                # Pinch detection + click
                pinch_dist = get_pinch_distance(tx, ty, thumb_x, thumb_y)
                cv2.putText(frame, f"PINCH: {int(pinch_dist)}px",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                click_cooldown_counter = handle_pinch_click(frame, pinch_dist, click_cooldown_counter)

                # ── GESTURE: THUMBS UP → OPEN SAFARI ──────────────
                if is_thumbs_up(hand):
                    thumbs_up_counter += 1
                    progress = min(thumbs_up_counter, THUMBS_UP_HOLD_FRAMES)
                    bar_width = int((progress / THUMBS_UP_HOLD_FRAMES) * 150)
                    cv2.rectangle(frame, (10, 220), (10 + bar_width, 235), (0, 255, 255), -1)
                    cv2.rectangle(frame, (10, 220), (160, 235), (255, 255, 255), 1)
                    cv2.putText(frame, "THUMBS UP - HOLD TO LAUNCH",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    if thumbs_up_counter >= THUMBS_UP_HOLD_FRAMES and app_launch_cooldown == 0:
                        open_app("Safari")
                        cv2.putText(frame, "WEBSHOOTER LAUNCH: SAFARI",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                        app_launch_cooldown = APP_LAUNCH_COOLDOWN
                        thumbs_up_counter = 0
                else:
                    thumbs_up_counter = 0

                # ── GESTURE: TWO-FINGER SCROLL ────────────────────
                if is_scroll_gesture(hand):
                    middle_tip = hand[12]
                    scroll_y = (tip.y + middle_tip.y) / 2  # average of index+middle y

                    cv2.putText(frame, "SCROLL MODE",
                        (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    if prev_scroll_y is not None:
                        delta_y = scroll_y - prev_scroll_y

                        if abs(delta_y) > SCROLL_DEADZONE:
                            scroll_amount = int(-delta_y * SCROLL_SENSITIVITY)
                            pyautogui.scroll(scroll_amount)

                            direction = "UP" if scroll_amount > 0 else "DOWN"
                            cv2.putText(frame, f"SCROLLING {direction}",
                                (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    prev_scroll_y = scroll_y
                else:
                    prev_scroll_y = None

                # ── CALIBRATION / CURSOR ──────────────────────────
                if calibrating:
                    cal, cal_frames, calibrating_done = run_calibration_step(
                        frame, tip, cal, cal_frames, frame_width, frame_height
                    )
                    if calibrating_done:
                        calibrating = False
                else:
                    # Only move the cursor if NOT in scroll mode
                    if not is_scroll_gesture(hand):
                        target_x, target_y = map_to_screen(tip, cal, screen_width, screen_height)
                        smooth_x, smooth_y = smooth_position(prev_x, prev_y, target_x, target_y, SMOOTHING)
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                        prev_x, prev_y = smooth_x, smooth_y

                    cv2.putText(frame, "SPIDER CURSOR ACTIVE",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    draw_hud_text(frame, get_hud_status_lines(pinch_dist))
        else:
            thumbs_up_counter = 0
            prev_scroll_y = None

        # Spider logo always visible
        draw_spider_logo(frame, frame_width - 50, 50, size=30)

        cv2.imshow("SPIDERSENSE - PETER PARKER PROTOCOL", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()


if __name__ == "__main__":
    main()