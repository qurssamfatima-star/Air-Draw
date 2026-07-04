
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"
COLORS = [("Eraser", (50, 50, 50)), ("Red", (0, 0, 255)), ("Green", (0, 255, 0)),
          ("Blue", (255, 0, 0)), ("Yellow", (0, 255, 255)), ("Clear", (30, 30, 30))]
BAR_H = 70

# --- set up the new HandLandmarker  ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=mp_vision.RunningMode.VIDEO,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
h, w = frame.shape[:2]
canvas = np.zeros((h, w, 3), dtype=np.uint8)

color, thickness, tool = (0, 0, 255), 8, "Red"
prev_pt = None
start_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]  # list of 21 landmarks, .x/.y normalized
        x, y = int(lm[8].x * w), int(lm[8].y * h)
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y

        if index_up and middle_up:  # selection mode
            prev_pt = None
            if y <= BAR_H:
                slot = min(x // (w // len(COLORS)), len(COLORS) - 1)
                tool, c = COLORS[slot]
                if tool == "Clear":
                    canvas[:] = 0
                else:
                    color = c
                    thickness = 40 if tool == "Eraser" else 8
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

        elif index_up:  # draw mode
            if y > BAR_H:
                if prev_pt:
                    cv2.line(canvas, prev_pt, (x, y), color, thickness, cv2.LINE_AA)
                prev_pt = (x, y)
            else:
                prev_pt = None
            cv2.circle(frame, (x, y), thickness // 2 + 2, color, -1)
        else:
            prev_pt = None
    else:
        prev_pt = None

    # merge canvas onto live frame
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    frame = cv2.add(frame, canvas)

    # toolbar
    slot_w = w // len(COLORS)
    for i, (name, c) in enumerate(COLORS):
        x0 = i * slot_w
        cv2.rectangle(frame, (x0, 0), (x0 + slot_w, BAR_H), c, -1)
        if name == tool:
            cv2.rectangle(frame, (x0 + 2, 2), (x0 + slot_w - 2, BAR_H - 2), (255, 255, 255), 2)
        cv2.putText(frame, name, (x0 + 8, BAR_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "[c] clear  [s] save  [q] quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
    elif key == ord('s'):
        cv2.imwrite(f"drawing_{int(time.time())}.png", canvas)
        print("Saved!")

cap.release()
cv2.destroyAllWindows()