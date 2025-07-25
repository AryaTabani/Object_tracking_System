import cv2
import numpy as np
from ultralytics import YOLO
import random


def detect_multiple_objects(frame, target_class_name):
    model = YOLO('yolov8n.pt')
    results = model(frame, verbose=False)

    detected_boxes = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            detected_class_name = model.names[class_id]

            if detected_class_name.lower() == target_class_name.lower():
                x1, y1, x2, y2 = box.xyxy[0]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                detected_boxes.append((x, y, w, h))

    if not detected_boxes:
        print(f"Detector could not find any '{target_class_name}' instances in the image.")
    else:
        print(f"Detector found {len(detected_boxes)} instance(s) of '{target_class_name}'.")

    return detected_boxes


if __name__ == '__main__':
    frame = cv2.imread('img.png')
    if frame is not None:
        bboxes = detect_multiple_objects(frame, 'person')

        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'Person {i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Multi-Detection Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not load img.png. Please provide a valid image.")
