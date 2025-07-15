import cv2
import argparse

def create_tracker(tracker_type):

    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test OpenCV Trackers")
    parser.add_argument("--tracker", type=str, default="KCF", help="Tracker type: CSRT, KCF, or MOSSE")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")

    args = parser.parse_args()

    tracker = create_tracker(args.tracker)
    video = cv2.VideoCapture(args.video)

    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        exit()
    bbox = cv2.selectROI("Select Object", frame, False)
    cv2.destroyWindow("Select Object")

    tracker.init(frame, bbox)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame, f"Tracker: {args.tracker}", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  
            break

    video.release()
    cv2.destroyAllWindows()