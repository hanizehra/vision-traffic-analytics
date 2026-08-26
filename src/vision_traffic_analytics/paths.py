from pathlib import Path


def create_ground_truth_path(video_path: Path) -> Path:
    """Create the matching ground-truth JSON path."""

    video_parts = video_path.parts

    try:
        videos_index = video_parts.index("videos")
    except ValueError:
        raise ValueError(
            "Video path must be inside the data/videos directory."
        )

    relative_parts = video_parts[videos_index + 1:]

    category = relative_parts[0]
    video_name = video_path.stem

    return (
        Path("data")
        / "ground_truth"
        / category
        / f"{video_name}.json"
    )