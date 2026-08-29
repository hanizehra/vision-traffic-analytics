import argparse
import json
from pathlib import Path

import cv2

from vision_traffic_analytics.config import VIDEO_PATHS
from vision_traffic_analytics.paths import create_ground_truth_path
from vision_traffic_analytics.frame_utils import prepare_display_frame, convert_to_display_coordinates


KEY_SPACE = ord(" ")
KEY_PREVIOUS = 97
KEY_NEXT = 100
KEY_UNDO = ord("u")     
KEY_IN = ord("i")
KEY_OUT = ord("o")
KEY_SAVE = ord("s")
KEY_ESC = 27

WINDOW_NAME = "Ground Truth Annotator"

LINE_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)


def load_ground_truth(file_path: Path) -> dict:
    """Load existing ground-truth data from JSON."""

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_ground_truth(
    file_path: Path,
    ground_truth: dict,
) -> None:
    """Save updated ground-truth data to JSON."""

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            ground_truth,
            file,
            indent=4,
        )

    print(
        f"Ground-truth saved to: {file_path}"
    )


def get_next_event_id(
    events: list[dict],
) -> int:
    """Return the next available event ID."""

    if not events:
        return 1

    return max(
        event["track_id"]
        for event in events
    ) + 1


def create_event(
    track_id: int,
    object_class: str,
    line_id: str,
    direction: str,
    frame_number: int,
    fps: float,
) -> dict:
    """Create a ground-truth crossing event."""

    timestamp_sec = frame_number / fps

    return {
        "track_id": track_id,
        "class": object_class,
        "line_id": line_id,
        "direction": direction,
        "frame": frame_number,
        "timestamp_sec": round(
            timestamp_sec,
            2,
        ),
    }


def get_object_class(
    video_name: str,
) -> str:
    """Return the object class for the selected video."""

    if video_name.startswith("c"):
        return "car"

    return "person"


def move_to_frame(
    video_capture: cv2.VideoCapture,
    frame_number: int,
) -> tuple[bool, object]:
    """Move to a specific frame and return it."""

    video_capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number,
    )

    success, frame = video_capture.read()

    return success, frame


def draw_line(
    image,
    line_points: list[list[int]],
) -> None:
    """Draw the counting line on the displayed frame."""

    point_1 = tuple(line_points[0])
    point_2 = tuple(line_points[1])

    cv2.line(
        image,
        point_1,
        point_2,
        LINE_COLOR,
        2,
    )


def draw_status(
    image,
    frame_number: int,
    fps: float,
    events: list[dict],
    pending_events: list[dict],
    is_paused: bool,
) -> None:
    """Draw annotation status information on the frame."""

    timestamp_sec = frame_number / fps

    state = "PAUSED" if is_paused else "PLAYING"

    cv2.putText(
        image,
        f"Frame: {frame_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
    )

    cv2.putText(
        image,
        f"Time: {timestamp_sec:.2f}s",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
    )

    cv2.putText(
        image,
        f"State: {state}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
    )

    cv2.putText(
        image,
        f"Events: {len(events)}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
    )

    if pending_events:

        pending_text = (
            f"Marked this frame: "
            f"{len(pending_events)}"
        )

        cv2.putText(
            image,
            pending_text,
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
        )


def print_event(
    event: dict,
) -> None:
    """Print a newly created annotation event."""

    print(
        f"[Frame {event['frame']} | "
        f"t={event['timestamp_sec']:.2f}s] "
        f"{event['direction'].upper()} marked "
        f"→ track_id={event['track_id']}"
    )

def undo_last_event(events: list[dict]) -> int | None:
    """Remove the most recently added event and return its ID."""

    if not events:
        print("No events to undo.")
        return None

    removed_event = events.pop()

    print(
        f"Undid event → track_id={removed_event['track_id']} "
        f"| frame={removed_event['frame']} "
        f"| direction={removed_event['direction']}"
    )

    return removed_event["track_id"]


def parse_arguments() -> argparse.Namespace:
    """Read the selected video name from the command line."""

    parser = argparse.ArgumentParser(
        description="Annotate ground-truth crossing events."
    )

    parser.add_argument(
        "video",
        choices=VIDEO_PATHS.keys(),
        help="Video name: c1, c2, p1, or p2",
    )

    return parser.parse_args()


def main() -> None:
    """Run the ground-truth annotation tool."""

    args = parse_arguments()

    video_path = Path(
        VIDEO_PATHS[args.video]
    )

    ground_truth_path = create_ground_truth_path(
        video_path
    )

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: "
            f"{ground_truth_path}"
        )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    fps = ground_truth["fps"]
    events = ground_truth["events"]

    object_class = get_object_class(
        args.video
    )

    line_points = ground_truth["lines"][0]["points"]
    line_id = ground_truth["lines"][0]["id"]

    next_event_id = get_next_event_id(events)

    is_paused = True
    current_frame_number = 0

    pending_events = []

    print(f"Video: {video_path}")
    print(f"Ground-truth: {ground_truth_path}")
    print(f"FPS: {fps}")
    print(f"Resolution: {ground_truth['resolution']}")
    print(f"Line: {line_points}")
    print(f"Existing events: {len(events)}")
    print(f"Object class: {object_class}")
    print(f"Next Event ID: {next_event_id}")

    video_capture = cv2.VideoCapture(str(video_path))

    if not video_capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    success, frame = video_capture.read()

    if not success:
        video_capture.release()

        raise RuntimeError("Could not read the first frame the frame ")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_AUTOSIZE,
    )

    while True:

        display_canvas, scale_factor, offset_x, offset_y = (
            prepare_display_frame(frame)
        )

        display_line_points = [
            convert_to_display_coordinates(
                point,
                scale_factor,
                offset_x,
                offset_y,
            )
            for point in line_points
        ]

        display_image = display_canvas.copy()

        cv2.line(
            display_image,
            display_line_points[0],
            display_line_points[1],
            LINE_COLOR,
            2,
        )

        draw_line(
            display_image,
            line_points,
        )

        draw_status(
            display_image,
            current_frame_number,
            fps,
            events,
            pending_events,
            is_paused,
        )

        cv2.imshow(
            WINDOW_NAME,
            display_image,
        )

        if is_paused:

            key = cv2.waitKeyEx(0)

        else:

            key = cv2.waitKeyEx(int(1000 / fps))

        if key == KEY_ESC:
            break

        if key == KEY_SPACE:

            is_paused = not is_paused

            continue

        if is_paused and key == KEY_PREVIOUS:

            if current_frame_number > 0:

                current_frame_number -= 1

                success, frame = move_to_frame(
                    video_capture,
                    current_frame_number,
                )

                if not success:
                    break

            pending_events.clear()

            continue

        if is_paused and key == KEY_NEXT:

            current_frame_number += 1

            success, frame = move_to_frame(
                video_capture,
                current_frame_number,
            )

            if not success:

                current_frame_number -= 1

                break

            pending_events.clear()

            continue

        if is_paused and key in (KEY_IN, KEY_OUT):

            if key == KEY_IN:
                direction = "in"
            else:
                direction = "out"

            event = create_event(
                next_event_id,
                object_class,
                line_id,
                direction,
                current_frame_number,
                fps,
            )

            events.append(event)

            print(
                f"[Frame {current_frame_number} | "
                f"t={event['timestamp_sec']:.2f}s] "
                f"{direction.upper()} marked "
                f"→ track_id={next_event_id}"
            )

            next_event_id += 1

        if is_paused and key == KEY_SAVE:

            save_ground_truth(
                ground_truth_path,
                ground_truth,
            )

            pending_events.clear()

            continue

        if not is_paused:

            success, frame = video_capture.read()

            if not success:
                break

            current_frame_number += 1

        if is_paused and key == KEY_UNDO:

            removed_event_id = undo_last_event(events)

            if removed_event_id is not None:
                next_event_id = removed_event_id        

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()