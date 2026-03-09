# Hand Shapes Neon (MediaPipe + OpenCV)

A real-time hand-gesture project built with MediaPipe Tasks + OpenCV.  
The webcam tracks one hand and lets you control neon shapes using gestures.

## Features
- Real-time hand landmark detection
- Gesture-based shape switching
  - Open palm (4-5 fingers) -> next shape
  - Peace sign (index + middle) -> previous shape
- Pinch-based dynamic size control (thumb-index distance)
- Neon visual styling
  - Camera frame stays normal
  - Hand landmarks in neon blue
  - Hand connections in lighter cyan
  - Optional glow overlay

## Project File
- `hand_shapes_neon.py`

## Requirements
- Python 3.10+
- `mediapipe`
- `opencv-python`
- `numpy`
- `hand_landmarker.task` model file in the same folder as script

## Installation
```bash
python3 -m pip install mediapipe opencv-python numpy
