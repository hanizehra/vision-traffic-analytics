import argparse
import json
from datetime import date
from pathlib import Path 
import cv2

from vision_traffic_analytics.config import VIDEO_PATHS

# predefined area used to display the video.
# video keeps its original aspect ratio inside this area.
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 600

KEY_ESC = 27
KEY_ENTER = 13
KEY_RESET = ord("r")

LINE_ID = "L1"
LINE_NAME = "count_line"


# stores the two points coordinates(x,y).
selected_points = []


def calculate_display_size(frame_width: int, frame_height:int) -> tuple[int,int,float]:
    """calculate the proportional display size of the video."""

    width_scale = DISPLAY_WIDTH / frame_width
    height_scale = DISPLAY_HEIGHT / frame_height

    # use the smaller scale so the entire frame fits without cropping.
    scale_factor = min(width_scale, height_scale)

    display_width = int(frame_width * scale_factor)
    display_height = int(frame_height * scale_factor)

    return display_width, display_height, scale_factor


def convert_to_original_coordinates(
    point : tuple[int,int],
    scale_factor: float,
    offset_x: int,
    offset_y: int,
) -> tuple [int,int]:
    
    """convert display coordinates back to original frame coordinates."""

    display_x, display_y = point

    # remove the empty space around the resized video.
    video_display_x = display_x - offset_x
    video_display_y = display_y - offset_y

    # reverse the scaling to get the original frame coordinates.
    original_x = int(video_display_x / scale_factor)
    original_y = int(video_display_y / scale_factor)

    return original_x, original_y


def handle_mouse_event(event : int, x : int, y : int, flags : int, parameters) -> None:
    """store a point when the user clicks inside the window."""

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # only two points are needed to define the line.
    if len(selected_points) >= 2:
        return

    selected_points.append((x, y))

    print(f"Display coordinate: ({x}, {y})")


def create_output_path(video_path: Path) -> Path:
    """Create the matching ground-truth JSON path."""

    video_parts = video_path.parts

    try:
        videos_index = video_parts.index("videos")
    except ValueError:
        raise ValueError(
            "Video path must be inside the data/videos directory."
        )

    relative_parts = video_parts[videos_index + 1 :]

    category = relative_parts[0]
    video_name = video_path.stem

    return (
        Path("data")
        / "ground_truth"
        / category
        / f"{video_name}.json"
    )


def save_line_data(
    video_path: Path,
    output_path: Path,
    frame_width: int,
    frame_height: int,
    fps: float,
    frame_count: int,
    original_points: list[tuple[int, int]],
) -> None:
    """Save video metadata and counting line to JSON."""

    duration_sec = frame_count / fps if fps > 0 else 0

    data = {
        "schema_version": "1.0",
        "video": video_path.as_posix(),
        "fps": fps,
        "resolution": [frame_width, frame_height],
        "duration_sec": round(duration_sec, 2),
        "lines": [
            {
                "id": LINE_ID,
                "name": LINE_NAME,
                "points": [
                    list(point)
                    for point in original_points
                ],
            }
        ],
        "annotator": "Hani",
        "annotation_date": date.today().isoformat(),
        "events": [],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    print(f"Ground-truth file saved to: {output_path}")



def get_video_path() -> Path:
    """Get the video path from the selected video name."""

    parser = argparse.ArgumentParser(
        description="Select a counting line for a video."
    )

    parser.add_argument(
        "video",
        choices=VIDEO_PATHS.keys(),
        help="Video name: c1, c2, p1, or p2",
    )

    args = parser.parse_args()

    return Path(VIDEO_PATHS[args.video])


def main() -> None:
    """Select a counting line and save its original coordinates."""

    video_path = get_video_path()

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    output_path = create_output_path(video_path)

    video_capture = cv2.VideoCapture(
        str(video_path)
    )

    if not video_capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # Read one frame because we only need it for line selection.
    success, frame = video_capture.read()

    if not success:
        video_capture.release()
        raise RuntimeError(
            "Could not read the first frame from the video."
        )

    # Get the original frame dimensions.
    frame_height, frame_width = frame.shape[:2]

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Original frame resolution: "
        f"{frame_width}x{frame_height}"
    )

    # Calculate the proportional display size.
    display_width, display_height, scale_factor = (
        calculate_display_size(
            frame_width,
            frame_height,
        )
    )

    print(
        f"Display video resolution: "
        f"{display_width}x{display_height}"
    )

    # Resize the frame while preserving its aspect ratio.
    display_frame = cv2.resize(
        frame,
        (display_width, display_height),
    )

    # Calculate the empty space around the resized video.
    offset_x = (DISPLAY_WIDTH - display_width) // 2
    offset_y = (DISPLAY_HEIGHT - display_height) // 2

    # Create the fixed-size display canvas.
    display_canvas = cv2.copyMakeBorder(
        display_frame,
        offset_y,
        DISPLAY_HEIGHT - display_height - offset_y,
        offset_x,
        DISPLAY_WIDTH - display_width - offset_x,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # Create the window and connect the mouse callback.
    cv2.namedWindow(
        "LineSelector",
        cv2.WINDOW_AUTOSIZE,
    )

    cv2.setMouseCallback(
        "LineSelector",
        handle_mouse_event,
    )

    while True:

        # Create a fresh copy so drawings do not modify the canvas.
        display_image = display_canvas.copy()

        # Show the first selected point.
        if len(selected_points) == 1:

            cv2.circle(
                display_image,
                selected_points[0],
                5,
                (0, 255, 0),
                -1,
            )

        # Show the line after both points are selected.
        elif len(selected_points) == 2:

            cv2.line(
                display_image,
                selected_points[0],
                selected_points[1],
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            "LineSelector",
            display_image,
        )

        # Process keyboard and mouse events.
        key = cv2.waitKey(1) & 0xFF

        # ESC → exit without saving.
        if key == KEY_ESC:
            break

        # R → remove the selected points.
        if key == KEY_RESET:
            selected_points.clear()
            print("Points reset.")

        # ENTER → confirm the line.
        if key == KEY_ENTER and len(selected_points) == 2:
            break

    # Convert selected points to original coordinates.
    if len(selected_points) == 2:

        original_points = [
            convert_to_original_coordinates(
                point,
                scale_factor,
                offset_x,
                offset_y,
            )
            for point in selected_points
        ]

        for index, point in enumerate(
            original_points,
            start=1,
        ):
            print(
                f"Original coordinate {index}: {point}"
            )

        save_line_data(
            video_path,
            output_path,
            frame_width,
            frame_height,
            fps,
            frame_count,
            original_points,
        )

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()