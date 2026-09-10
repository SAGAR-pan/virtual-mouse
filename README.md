# 🖱️ Hand Gesture Virtual Mouse

A Python-based virtual mouse that allows users to control their computer cursor and perform mouse actions using hand gestures captured through a webcam.

## 📌 Project Overview

This project uses computer vision and hand tracking to detect hand movements and convert them into mouse actions.

The webcam captures the user's hand, MediaPipe detects hand landmarks, and PyAutoGUI translates the detected gestures into cursor movement, clicking, and scrolling.

## ✨ Features

- 🖐️ Real-time hand tracking
- 🖱️ Cursor movement using the index finger
- 👆 Left-click using thumb + index finger
- 🖱️ Right-click using thumb + middle finger
- 🔄 Two-finger scrolling
- 🎯 Cursor smoothing for better control
- 📷 Live webcam feed
- 🛑 Safety fail-safe for mouse control

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – Webcam and image processing
- **MediaPipe** – Hand landmark detection
- **PyAutoGUI** – Mouse and screen control

## ⚙️ How It Works

Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Tracking
   ↓
Hand Landmark Detection
   ↓
Gesture Recognition
   ↓
PyAutoGUI
   ↓
Mouse Actions


## 🖐️ Gesture Controls

| Gesture | Action |
|---|---|
| Move index finger | Move cursor |
| Thumb + index finger | Left click |
| Thumb + middle finger | Right click |
| Index + middle finger | Scroll |

## 🚀 Installation
 1. Clone the repository

```bash
git clone https://github.com/SAGAR-pan/virtual-mouse.git
cd virtual-mouse

2. Create a virtual environment
python -m venv venv

3. Activate the virtual environment
venv\Scripts\activate

4. Install dependencies
pip install opencv-python mediapipe==0.10.21 pyautogui

5. Run the project
python virtual_mouse.py

## 🎥 Usage
Connect a working webcam.
Run virtual_mouse.py.
Place your hand in front of the webcam.
Use the supported gestures to control the mouse.
Press Q to exit.

## 📁 Project Structure
virtual-mouse/
│
├── virtual_mouse.py
├── .gitignore
└── README.md

## 👨‍💻 Author

Sagar B S

## GitHub: https://github.com/SAGAR-pan
```text
