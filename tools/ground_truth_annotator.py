import argparse
import json
from pathlib import Path

import cv2

from vision_traffic_analytics.config import VIDEO_PATHS
from vision_traffic_analytics.paths import create_ground_truth_path


def load_ground_truth(file_path: Path) -> dict:
    """Load existing ground-truth data from JSON."""

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
    """Load the selected video and its ground-truth data."""

    args = parse_arguments()

    video_path = Path(VIDEO_PATHS[args.video])
    ground_truth_path = create_ground_truth_path(video_path)

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {ground_truth_path}"
        )

    ground_truth = load_ground_truth(ground_truth_path)

    print(f"Video: {video_path}")
    print(f"Ground-truth: {ground_truth_path}")
    print(f"FPS: {ground_truth['fps']}")
    print(f"Resolution: {ground_truth['resolution']}")
    print(f"Line: {ground_truth['lines'][0]['points']}")
    print(f"Existing events: {len(ground_truth['events'])}")


if __name__ == "__main__":
    main()