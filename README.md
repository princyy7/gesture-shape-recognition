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
