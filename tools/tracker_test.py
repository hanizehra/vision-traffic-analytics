import argparse
import json
from pathlib import Path

import cv2

from vision_traffic_analytics.tracking.tracker import ObjectTracker
from vision_traffic_analytics.counting.counter import (
    Counter,
    load_counting_line,
)

from vision_traffic_analytics.config import (
    VIDEO_PATHS,
    GROUND_TRUTH_PATHS,
    PREDICTION_PATHS,
    COUNTING_CLASSES,
    IN_TRANSITIONS,
)

from vision_traffic_analytics.frame_utils import (
    prepare_display_frame,
    convert_to_display_coordinates,
)


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
        help="Video name: v1, v2, p1, or p2",
    )

    return parser.parse_args()


def draw_counting_line(
    frame,
    counting_line,
    scale_factor: float,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw the counting line on the display frame."""

    start = convert_to_display_coordinates(
        counting_line.start,
        scale_factor,
        offset_x,
        offset_y,
    )

    end = convert_to_display_coordinates(
        counting_line.end,
        scale_factor,
        offset_x,
        offset_y,
    )

    cv2.line(
        frame,
        start,
        end,
        (255, 0, 0),
        2,
    )


def draw_tracks(
    frame,
    tracks,
    scale_factor: float,
    offset_x: int,
    offset_y: int,
) -> None:
    """Draw tracking labels and center points."""

    for track in tracks:

        center_x, center_y = track.center

        display_center = convert_to_display_coordinates(
            [center_x, center_y],
            scale_factor,
            offset_x,
            offset_y,
        )

        label = (
            f"{track.class_name} "
            f"ID:{track.track_id} "
            f"{track.confidence:.2f}"
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        label_x = display_center[0]
        label_y = max(
            display_center[1] - 10,
            text_height + baseline + 5,
        )

        cv2.rectangle(
            frame,
            (
                label_x,
                label_y - text_height - baseline,
            ),
            (
                label_x + text_width,
                label_y,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            label,
            (label_x, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.circle(
            frame,
            display_center,
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

    ground_truth_path = GROUND_TRUTH_PATHS[args.video]

    counting_line = load_counting_line(
        ground_truth_path
    )

    counter = Counter(
        counting_line,
        IN_TRANSITIONS[args.video],
    )

    counting_class = COUNTING_CLASSES[args.video]

    fps = video_capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_width = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    is_paused = False
    frame_number = 0
    prediction_events = []

    while True:

        if not is_paused:

            success, frame = video_capture.read()

            if not success:
                break

            frame_number += 1

            tracks = tracker.track(frame)

            for track in tracks:

                if track.class_name != counting_class:
                    continue

                direction = counter.update(
                    track.track_id,
                    track.center,
                )

                if direction:

                    print(
                        f"Track ID {track.track_id}: {direction}"
                    )

                    prediction_events.append(
                        {
                            "track_id": track.track_id,
                            "class": track.class_name,
                            "line_id": "L1",
                            "direction": direction,
                            "frame": frame_number,
                            "timestamp_sec": round(
                                frame_number / fps,
                                2,
                            ),
                        }
                    )

            display_frame, scale_factor, offset_x, offset_y = (
                prepare_display_frame(frame)
            )

            draw_counting_line(
                display_frame,
                counting_line,
                scale_factor,
                offset_x,
                offset_y,
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

            cv2.putText(
                display_frame,
                f"IN: {counter.in_count}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"OUT: {counter.out_count}",
                (20, 100),
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

    prediction_path = PREDICTION_PATHS[args.video]

    prediction_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_data = {
        "schema_version": "1.0",
        "video": video_path.as_posix(),
        "fps": fps,
        "resolution": [
            frame_width,
            frame_height,
        ],
        "duration_sec": round(
            total_frames / fps,
            2,
        ),
        "lines": [
            {
                "id": "L1",
                "name": "count_line",
                "points": [
                    list(counting_line.start),
                    list(counting_line.end),
                ],
            }
        ],
        "annotator": "YOLO26s + ByteTrack",
        "annotation_date": None,
        "events": prediction_events,
    }

    with prediction_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            prediction_data,
            file,
            indent=4,
        )

    print(
        f"Prediction saved to: {prediction_path}"
    )

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()