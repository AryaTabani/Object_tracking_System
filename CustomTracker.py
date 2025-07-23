import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import time

video_path = "videos/person4.mp4"
output_path = "person4/person4_output.mp4"

model = YOLO("yolov8n.pt")


def create_kalman(cx, cy):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 50
    kf.R *= 10
    kf.Q = np.eye(4)
    kf.x[:2] = np.array([[cx], [cy]])
    return kf


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video at {video_path}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_source = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_source, (frame_width, frame_height))

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read the first frame")

results = model(frame, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()
scores = results[0].boxes.conf.cpu().numpy()
if len(boxes) == 0:
    raise RuntimeError("No objects detected in the first frame")

best_idx = int(np.argmax(scores))
x1, y1, x2, y2 = boxes[best_idx]
cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
w, h = x2 - x1, y2 - y1

kf = create_kalman(cx, cy)

tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(frame, (x1, y1, w, h))

fps_log = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    ok, bbox = tracker.update(frame)

    if ok:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        kf.update(np.array([cx, cy]))
    else:
        kf.predict()
        cx, cy = kf.x[0].item(), kf.x[1].item()
        x, y = cx - w / 2, cy - h / 2
        bbox = (x, y, w, h)

    kf.predict()

    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

    processing_fps = 1.0 / (time.time() - start)
    fps_log.append(processing_fps)
    cv2.putText(frame, f"FPS: {processing_fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow("Custom Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved to {output_path}")
print("Average Processing FPS:", np.mean(fps_log))