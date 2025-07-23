import argparse

import cv2

from baseline_trackers import create_tracker
from detector import detect_initial_object
from baseline_trackers import create_tracker
from CustomTracker import CsrtKalmanTracker # Import the new tracker
def main(args):
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    output_filename = f"output_{args.tracker}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    ok, frame = video.read()
    if not ok:
        print("Error: Could not read first frame.")
        return

    bbox = detect_initial_object(frame, args.target_class)
    if bbox is None:
        print(f"Could not detect '{args.target_class}'. Exiting.")
        bbox = cv2.selectROI("Manual Selection", frame, False)
        cv2.destroyWindow("Manual Selection")
        if not bbox[2] or not bbox[3]: return

    if args.tracker == 'CUSTOM':
        tracker = CsrtKalmanTracker()
    else:
        tracker = create_tracker(args.tracker)

    tracker.init(frame, bbox)

    frame_count = 0
    total_fps = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()

        if args.tracker == 'CUSTOM':
            success, bbox = tracker.update(frame)
        else:
            success, bbox = tracker.update(frame)

        frame_fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        total_fps += frame_fps
        frame_count += 1

        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking Failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        cv2.putText(frame, f"Tracker: {args.tracker}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(frame_fps)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    avg_fps = total_fps / frame_count if frame_count > 0 else 0
    print(f"Tracker: {args.tracker} | Average FPS: {avg_fps:.2f}")

    video.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end object tracking system.")
    parser.add_argument("--video", type=str, default="videos/person3.mp4", help="Path to the input video.")
    parser.add_argument("--tracker", type=str, default="KCF", help="Tracker type: CSRT, KCF, MOSSE, CUSTOM.")
    parser.add_argument("--target_class", type=str, default="person",
                        help="The object class to track (e.g., 'person', 'car').")

    args = parser.parse_args()
    main(args)
