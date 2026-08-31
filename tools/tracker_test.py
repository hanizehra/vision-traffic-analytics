import argparse
from pathlib import Path

import cv2

from vision_traffic_analytics.config import VIDEO_PATHS
from vision_traffic_analytics.frame_utils import (
    prepare_display_frame,
    convert_to_display_coordinates,
)
from vision_traffic_analytics.tracking.tracker import ObjectTracker


KEY_SPACE = ord(" ")
KEY_ESC = 27

WINDOW_NAME = "ByteTrack Tracking Test"


def parse_arguments() -> argparse.Namespace:
    """Read the selected video name from the command line."""

    parser = argparse.ArgumentParser(
        description="Test YOLO26 detection with ByteTrack tracking."
    )

    parser.add_argument(
        "video",
        choices=VIDEO_PATHS.keys(),
        help="Video name: c1, c2, p1, or p2",
    )

    return parser.parse_args()


def draw_tracks(
    frame,
    tracks,
    scale_factor: float,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw tracking bounding boxes, labels, and track IDs."""

    for track in tracks:

        x1, y1, x2, y2 = track.bounding_box
        center_x, center_y = track.center

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
            f"{track.class_name} "
            f"ID:{track.track_id} "
            f"{track.confidence:.2f}"
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
                min(
                    y2_display + 20,
                    frame.shape[0] - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.circle(
            frame,
            (
                int(center_x * scale_factor + offset_x),
                int(center_y * scale_factor + offset_y),
            ),
            4,
            (255, 0, 0),
            -1,
        )


def main() -> None:
    """Run YOLO26 with ByteTrack on the selected video."""

    args = parse_arguments()

    video_path = Path(VIDEO_PATHS[args.video])

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    video_capture = cv2.VideoCapture(
        str(video_path)
    )

    if not video_capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    tracker = ObjectTracker()

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

            tracks = tracker.track(frame)

            display_frame, scale_factor, offset_x, offset_y = (
                prepare_display_frame(frame)
            )

            draw_tracks(
                display_frame,
                tracks,
                scale_factor,
                offset_x,
                offset_y,
            )

            cv2.putText(
                display_frame,
                f"Tracks: {len(tracks)}",
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