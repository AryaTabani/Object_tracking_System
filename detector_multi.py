from ultralytics import YOLO


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
        print(f"Detector could not find any '{target_class_name}' in the frame.")
    else:
        print(f"Detector found {len(detected_boxes)} instance(s) of '{target_class_name}'.")

    return detected_boxes