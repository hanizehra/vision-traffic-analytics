import argparse
from pathlib import Path

import cv2


from vision_traffic_analytics.detection.detector import ObjectDetector
from vision_traffic_analytics.frame_utils import (prepare_display_frame, convert_to_display_coordinates) 
from vision_traffic_analytics.config import VIDEO_PATHS

KEY_SPACE = ord(" ")
KEY_ESC = 27

WINDOW_NAME = "YOLO26 Detection Test"


def parse_arguments() -> argparse.Namespace:
    """Read the video path from the command line."""

    parser = argparse.ArgumentParser(
        description="Test YOLO26 object detection on a video."
    )

    parser.add_argument(
        "video",
        choices=VIDEO_PATHS.keys(),
        help="Video name: v1, v2, p1, p2",
    )

    return parser.parse_args()


def draw_detections(
    frame,
    detections,
    scale_factor: float,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw detection bounding boxes and labels."""

    for detection in detections:

        x1, y1, x2, y2 = detection.bounding_box

        display_top_left = convert_to_display_coordinates(
            [x1, y1],
            scale_factor,
            offset_x,
            offset_y,
        )

        display_bottom_right = convert_to_display_coordinates(
            [x2, y2],
            scale_factor,
            offset_x,
            offset_y,
        )

        x1_display, y1_display = display_top_left
        x2_display, y2_display = display_bottom_right

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1_display, y1_display),
            (x2_display, y2_display),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (
                x1_display,
                min(y2_display + 20, frame.shape[0] - 5),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

def main() -> None:
    """Run YOLO26 detection on the selected video."""

    args = parse_arguments()

    video_path = Path(
         VIDEO_PATHS[args.video]
    )

    video_capture = cv2.VideoCapture(
        str(video_path)
    )

    if not video_capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    detector = ObjectDetector()

    is_paused = False

    frame_number = 0
    max_frames = 100

    while True:

        if not is_paused:

            success, frame = video_capture.read()

            if not success:
                break

            frame_number += 1

            if frame_number > max_frames:
                break

            detections = detector.detect(frame)

            display_frame, scale_factor, offset_x, offset_y = (
                prepare_display_frame(frame)
            )

            draw_detections(
                display_frame,
                detections,
                scale_factor,
                offset_x,
                offset_y,
            )

            cv2.putText(
                display_frame,
                f"Detections: {len(detections)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == KEY_ESC:
            break

        if key == KEY_SPACE:
            is_paused = not is_paused

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()