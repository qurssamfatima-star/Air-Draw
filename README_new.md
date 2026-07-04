# Air Canvas — Draw in the Air with Hand Gestures (New MediaPipe Tasks API)

Real-time computer vision app that turns your webcam + index finger into a
paintbrush. Built with **OpenCV** and **MediaPipe Tasks (HandLandmarker)**.

> **Note:** Recent MediaPipe releases (0.10.30+) removed the old
> `mp.solutions.hands` API on many platforms, including Windows. This
> version uses the current, actively-supported **Tasks API**
> (`mediapipe.tasks.python.vision.HandLandmarker`) instead.

## How it works

MediaPipe's HandLandmarker model detects 21 landmarks per hand every frame.
The app checks fingertip vs. PIP-joint y-position to decide which fingers
are "up," and maps that to a mode:

| Gesture | Mode | Behavior |
|---|---|---|
| Index finger only | **Draw** | Fingertip paints a line on the canvas |
| Index + middle finger | **Select** | Hover over the top toolbar to pick a color/tool (no drawing) |
| Fist / other | **Idle** | Cursor tracked, nothing happens |

The canvas is a separate layer composited onto the live camera feed each
frame, so drawings persist as your hand moves.

## Setup

**1. Create and activate a virtual environment (Python 3.10 or 3.11 recommended):**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
```

**2. Install dependencies:**
```bash
pip install -r requirements_new_api.txt
```

**3. Download the hand landmark model** (required — this file is not
bundled with the pip package):
```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
If `curl` isn't available (some Windows setups), paste that URL directly
into a browser and save the downloaded file as `hand_landmarker.task` in
the same folder as the script.

**4. Run:**
```bash
python air_canvas_new_api.py
```

Requires a working webcam.

## Controls

- **Index finger up** → draw
- **Index + middle finger up** → hover toolbar to change color/tool
- **`c`** → clear canvas
- **`s`** → save current drawing as `drawing_<timestamp>.png`
- **`q`** → quit

## Project structure

```
air_canvas/
├── air_canvas_new_api.py     # main application (Tasks API)
├── requirements_new_api.txt
├── hand_landmarker.task      # downloaded model (you add this)
└── drawing_*.png             # saved drawings (created when you press 's')
```

## Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
You're running an old script that uses the legacy API. Use
`air_canvas_new_api.py`, not a version that calls `mp.solutions.hands`.

**`RuntimeError` / model not found**
Make sure `hand_landmarker.task` is in the same folder as the script and
the filename is spelled exactly right.

**Webcam doesn't open**
Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have
multiple cameras, and confirm no other app is using the webcam.

## Possible extensions (good for a resume bullet / v2)

- HSV color-picker wheel instead of fixed palette
- Gesture-based brush thickness (e.g. thumb–index distance)
- Shape recognition (circle/line/rectangle) over raw strokes
- Streamlit/Gradio web version using `streamlit-webrtc` for a shareable demo link
- Two-hand support: one hand draws, other selects tools

## Tech stack

`Python` · `OpenCV` · `MediaPipe Tasks (HandLandmarker)` · `NumPy`
