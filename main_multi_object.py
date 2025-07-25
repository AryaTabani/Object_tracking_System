import cv2
import argparse
import random
from detector_multi import detect_multiple_objects
from baseline_trackers import create_tracker



def main(args):
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    output_filename = f"output_multi_{args.tracker}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    ok, frame = video.read()
    if not ok:
        print("Error: Could not read first frame.")
        return

    initial_bboxes = detect_multiple_objects(frame, args.target_class)
    if not initial_bboxes:
        print(f"Could not detect '{args.target_class}'. Exiting.")
        return

    trackers = []
    colors = []
    for bbox in initial_bboxes:
        if args.tracker == 'CUSTOM':
            tracker = HybridCorrelationKalmanTracker()
        else:
            tracker = create_tracker(args.tracker)

        tracker.init(frame, bbox)
        trackers.append(tracker)
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    frame_count = 0
    total_fps = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()

        for i, tracker in enumerate(trackers):
            if args.tracker == 'CUSTOM':
                success, bbox = tracker.update(frame)
            else:
                success, bbox = tracker.update(frame)

            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
                cv2.putText(frame, f"ID {i + 1}", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        frame_fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        total_fps += frame_fps
        frame_count += 1

        cv2.putText(frame, f"Tracker: {args.tracker}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(frame_fps)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(frame, f"Tracking {len(trackers)} objects", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0),
                    2)

        cv2.imshow("Multi-Object Tracking", frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    avg_fps = total_fps / frame_count if frame_count > 0 else 0
    print(f"Tracker: {args.tracker} | Average FPS: {avg_fps:.2f} for {len(trackers)} objects.")

    video.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-object tracking system.")
    parser.add_argument("--video", type=str, default="videos/person2.mp4", help="Path to the input video.")
    parser.add_argument("--tracker", type=str, default="KCF", help="Tracker type: CSRT, KCF, MOSSE, CUSTOM.")
    parser.add_argument("--target_class", type=str, default="person",
                        help="The object class to track (e.g., 'person', 'car').")

    args = parser.parse_args()
    main(args)