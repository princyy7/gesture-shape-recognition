import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "hand_landmarker.task"
SHAPES = ["circle", "square", "triangle", "star", "pentagon"]

# OpenCV uses BGR
SHAPE_COLOR = (255, 170, 40)          # neon bluish
LANDMARK_COLOR = (255, 120, 20)       # neon blue points
CONNECTION_COLOR = (255, 255, 140)    # lighter cyan lines
TEXT_COLOR = (240, 240, 240)

SIZE_SENSITIVITY = 2.2
MIN_SIZE = 8
MAX_SIZE_RATIO = 0.60
SWITCH_COOLDOWN = 0.8
ENABLE_GLOW = True

# Hand connection pairs (instead of mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


def draw_shape(frame, shape, center, size, color):
    x, y = center
    if shape == "circle":
        cv2.circle(frame, (x, y), size, color, -1)
    elif shape == "square":
        cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, -1)
    elif shape == "triangle":
        pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
        cv2.fillPoly(frame, [pts], color)
    elif shape == "star":
        pts = np.array([
            [x, y - size],
            [x + int(size * 0.3), y - int(size * 0.3)],
            [x + size, y],
            [x + int(size * 0.3), y + int(size * 0.3)],
            [x, y + size],
            [x - int(size * 0.3), y + int(size * 0.3)],
            [x - size, y],
            [x - int(size * 0.3), y - int(size * 0.3)]
        ], np.int32)
        cv2.fillPoly(frame, [pts], color)
    elif shape == "pentagon":
        pts = []
        for i in range(5):
            theta = np.deg2rad(i * 72 - 90)
            pts.append([int(x + size * np.cos(theta)), int(y + size * np.sin(theta))])
        cv2.fillPoly(frame, [np.array(pts, np.int32)], color)


def finger_states(hand_landmarks, hand_label):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    states = [0, 0, 0, 0, 0]

    thumb_tip = hand_landmarks[tips[0]]
    thumb_ip = hand_landmarks[pips[0]]
    if hand_label == "Right":
        states[0] = 1 if thumb_tip.x < thumb_ip.x else 0
    else:
        states[0] = 1 if thumb_tip.x > thumb_ip.x else 0

    for i in range(1, 5):
        tip = hand_landmarks[tips[i]]
        pip = hand_landmarks[pips[i]]
        states[i] = 1 if tip.y < pip.y else 0

    return states


def draw_hand_overlay(frame, hand_landmarks):
    h, w = frame.shape[:2]

    for i, j in HAND_CONNECTIONS:
        x1, y1 = int(hand_landmarks[i].x * w), int(hand_landmarks[i].y * h)
        x2, y2 = int(hand_landmarks[j].x * w), int(hand_landmarks[j].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), CONNECTION_COLOR, 2)

    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, LANDMARK_COLOR, -1)


def apply_glow_overlay(frame):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.circle(overlay, (w // 2, h // 2), int(min(w, h) * 0.45), (255, 220, 120), -1)
    blur = cv2.GaussianBlur(overlay, (0, 0), 35)
    return cv2.addWeighted(frame, 0.88, blur, 0.12, 0)


# -------------------------------
# Init
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check macOS camera permissions.")

cv2.namedWindow("Hand Shapes", cv2.WINDOW_NORMAL)

shape_index = 0
shape_size = 70
last_switch_time = 0.0


# -------------------------------
# Main loop
# -------------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    gesture_text = "No hand"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        hand_label = "Right"
        if result.handedness and result.handedness[0]:
            hand_label = result.handedness[0][0].category_name

        draw_hand_overlay(frame, hand)

        states = finger_states(hand, hand_label)
        open_count = sum(states)

        now = time.time()
        if now - last_switch_time > SWITCH_COOLDOWN:
            if open_count >= 4:
                shape_index = (shape_index + 1) % len(SHAPES)
                last_switch_time = now
                gesture_text = "Open palm -> NEXT"
            elif states == [0, 1, 1, 0, 0]:
                shape_index = (shape_index - 1) % len(SHAPES)
                last_switch_time = now
                gesture_text = "Peace sign -> PREV"

        thumb_tip = hand[4]
        index_tip = hand[8]
        px1, py1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        px2, py2 = int(index_tip.x * w), int(index_tip.y * h)

        pinch_dist = np.hypot(px2 - px1, py2 - py1)
        dynamic_max = int(min(w, h) * MAX_SIZE_RATIO)
        raw_size = pinch_dist * SIZE_SENSITIVITY
        shape_size = int(np.clip(raw_size, MIN_SIZE, dynamic_max))

        cv2.circle(frame, (px1, py1), 7, LANDMARK_COLOR, -1)
        cv2.circle(frame, (px2, py2), 7, LANDMARK_COLOR, -1)
        cv2.line(frame, (px1, py1), (px2, py2), CONNECTION_COLOR, 2)

        if gesture_text == "No hand":
            gesture_text = "Pinch controls size"

    draw_shape(frame, SHAPES[shape_index], center, shape_size, SHAPE_COLOR)

    if ENABLE_GLOW:
        frame = apply_glow_overlay(frame)

    cv2.putText(frame, f"Shape: {SHAPES[shape_index]}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)
    cv2.putText(frame, f"Size: {shape_size}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)
    cv2.putText(frame, gesture_text, (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 255, 220), 2)
    cv2.putText(frame, "q = quit", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    cv2.imshow("Hand Shapes", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

