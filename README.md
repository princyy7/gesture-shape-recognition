# Hand Shapes Neon

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-0097A7)
![OpenCV](https://img.shields.io/badge/OpenCV-computer%20vision-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

**Control glowing neon shapes with nothing but your hand.**

A real-time gesture project built with MediaPipe Tasks and OpenCV. Your webcam tracks one hand, and simple gestures let you switch shapes and resize them live.

---

## Features

- **Real-time hand tracking:** detects 21 hand landmarks per frame with the MediaPipe HandLandmarker (one hand).
- **Gesture shape switching:** open palm for the next shape, peace sign for the previous one.
- **Pinch to resize:** the thumb-to-index distance controls the shape's size.
- **Five shapes:** circle, square, triangle, star and pentagon.
- **Neon look:** neon-blue landmarks, cyan connection lines and an optional glow overlay.
- **Live HUD:** shows the current shape, its size and the detected gesture.
- **Easy to tweak:** colors, sensitivity and cooldown are constants at the top of the script.

---

## Controls

| Gesture | Action | How it's detected |
|:--------|:-------|:------------------|
| Open palm | Next shape | 4 or more fingers extended |
| Peace sign | Previous shape | Only index and middle fingers extended |
| Pinch | Resize the shape | Distance between thumb tip and index tip |
| `q` key | Quit | Closes the window and releases the camera |

**Shape order:** circle → square → triangle → star → pentagon (then back to circle).

**Size:** pinch distance × `2.2`, kept between `8` px and 60% of the frame's smaller side.

A short cooldown (`0.8 s`) after each switch stops the shape from flipping repeatedly while you hold a gesture.

---

## How It Works

1. **Capture** a frame from the webcam with OpenCV.
2. **Convert** it from BGR to RGB and pass it to the MediaPipe HandLandmarker.
3. **Read** the 21 landmarks and decide which fingers are up:
   - the thumb is checked by its horizontal position, using the detected handedness;
   - the other four fingers count as up when the tip is above its middle joint.
4. **Match** the finger pattern to a gesture and update the shape (with the cooldown).
5. **Measure** the pinch distance and convert it to the shape size.
6. **Draw** the hand skeleton, the shape and the glow overlay.
7. **Show** the HUD text and display the frame.

---

## Getting Started

### Requirements

- Python 3.10+
- A webcam
- `mediapipe`, `opencv-python`, `numpy`
- The `hand_landmarker.task` model file

### Installation

```bash
git clone https://github.com/princyy7/gesture-shape-recognition.git
cd gesture-shape-recognition
python3 -m pip install mediapipe opencv-python numpy
```

### Get the model

Download `hand_landmarker.task` from the [MediaPipe Hand Landmarker page](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and place it in the same folder as the script.

```text
gesture-shape-recognition/
├── hand_shapes_neon.py
├── hand_landmarker.task
└── README.md
```

### Run

```bash
python3 hand_shapes_neon.py
```

Hold up one hand in front of the camera. Press **`q`** to quit.

---

## Configuration

All settings are constants at the top of `hand_shapes_neon.py`.

| Setting | Default | What it does |
|:--------|:--------|:-------------|
| `SHAPES` | circle, square, triangle, star, pentagon | Shapes in the cycle |
| `SIZE_SENSITIVITY` | `2.2` | How strongly pinch distance scales the size |
| `MIN_SIZE` | `8` | Smallest shape size in pixels |
| `MAX_SIZE_RATIO` | `0.60` | Largest size as a fraction of the frame's smaller side |
| `SWITCH_COOLDOWN` | `0.8` | Seconds before another gesture can switch the shape |
| `ENABLE_GLOW` | `True` | Turns the glow overlay on or off |
| `SHAPE_COLOR` | `(255, 170, 40)` | Shape color |
| `LANDMARK_COLOR` | `(255, 120, 20)` | Landmark point color |
| `CONNECTION_COLOR` | `(255, 255, 140)` | Hand skeleton line color |
| `TEXT_COLOR` | `(240, 240, 240)` | HUD text color |

> Colors use OpenCV's **BGR** order, not RGB.
