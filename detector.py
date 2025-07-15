import cv2
from ultralytics import YOLO


def detect_initial_object(frame, target_class_name):
    model = YOLO('yolov8n.pt')
    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:

            class_id = int(box.cls[0])
            detected_class_name = model.names[class_id]

            if detected_class_name.lower() == target_class_name.lower():
                x1, y1, x2, y2 = box.xyxy[0]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                print(f"Detector found '{target_class_name}' at [{x}, {y}, {w}, {h}]")
                return (x, y, w, h)

    print(f"Detector could not find '{target_class_name}' in the first frame.")
    return None


if __name__ == '__main__':
    # sample
    frame = cv2.imread('sample_image.jpg')
    if frame is not None:
        bbox = detect_initial_object(frame, 'bicycle')
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Detected Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Detection Test", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Could not load sample_image.jpg. Please provide a valid image.")
