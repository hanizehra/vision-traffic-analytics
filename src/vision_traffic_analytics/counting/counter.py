import json
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class CountingLine:
    """Represent a counting line."""

    start: tuple[int, int]
    end: tuple[int, int]


class Counter:
    """Detect when tracked objects cross the counting line."""

    def __init__(
        self,
        counting_line: CountingLine,
        in_transition: tuple[int, int],
    ) -> None:
        self.counting_line = counting_line
        self.in_transition = in_transition

        self.previous_positions: dict[int, tuple[int, int]] = {}
        self.previous_sides: dict[int, int] = {}

        self.in_count = 0
        self.out_count = 0

    def update(
        self,
        track_id: int,
        center: tuple[int, int],
    ) -> str | None:
        """Update a track and return its crossing direction."""

        previous_position = self.previous_positions.get(track_id)
        previous_side = self.previous_sides.get(track_id)

        current_side = self._get_side(center)

        self.previous_positions[track_id] = center

        if current_side == 0:
            return None

        self.previous_sides[track_id] = current_side

        if previous_position is None or previous_side is None:
            return None

        if previous_side == current_side:
            return None

        if not self._segments_intersect(
            previous_position,
            center,
            self.counting_line.start,
            self.counting_line.end,
        ):
            return None

        transition = (previous_side, current_side)

        if transition == self.in_transition:
            self.in_count += 1
            return "in"

        self.out_count += 1
        return "out"

    def _get_side(
        self,
        point: tuple[int, int],
    ) -> int:
        """Return which side of the counting line a point is on."""

        x1, y1 = self.counting_line.start
        x2, y2 = self.counting_line.end

        px, py = point

        value = (
            (x2 - x1) * (py - y1)
            - (y2 - y1) * (px - x1)
        )

        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0

    @staticmethod
    def _orientation(
        a: tuple[int, int],
        b: tuple[int, int],
        c: tuple[int, int],
    ) -> int:
        """Return orientation of three points."""

        value = (
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )

        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0

    @classmethod
    def _segments_intersect(
        cls,
        a: tuple[int, int],
        b: tuple[int, int],
        c: tuple[int, int],
        d: tuple[int, int],
    ) -> bool:
        """Return whether two line segments intersect."""

        orientation_1 = cls._orientation(a, b, c)
        orientation_2 = cls._orientation(a, b, d)
        orientation_3 = cls._orientation(c, d, a)
        orientation_4 = cls._orientation(c, d, b)

        if orientation_1 != orientation_2 and orientation_3 != orientation_4:
            return True

        return False


def load_counting_line(
    ground_truth_path: Path,
) -> CountingLine:
    """Load the counting line from a ground-truth JSON file."""

    with ground_truth_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    lines = data["lines"]

    if not lines:
        raise ValueError(
            f"No counting line found in: {ground_truth_path}"
        )

    points = lines[0]["points"]

    if len(points) != 2:
        raise ValueError(
            f"Counting line must contain exactly two points: "
            f"{ground_truth_path}"
        )

    return CountingLine(
        start=tuple(points[0]),
        end=tuple(points[1]),
    )