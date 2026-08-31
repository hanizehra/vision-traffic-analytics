from dataclasses import dataclass
from ultralytics import YOLO

from vision_traffic_analytics.detection.detector import get_application_class


@dataclass
class Track:
    """Represent a tracked object."""

    track_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        """Return the center point of the bounding box."""

        x1, y1, x2, y2 = self.bounding_box

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )


class ObjectTracker:
    """Track objects using ByteTrack."""

    def __init__(
        self,
        model_path: str = "yolo26s.pt",
        confidence_threshold: float = 0.25,
    ) -> None:

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def track(self, frame) -> list[Track]:
        """Run YOLO26 detection with ByteTrack on a single frame."""

        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            imgsz=640,
            device="cpu",
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        tracks: list[Track] = []

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return tracks

        for box, track_id in zip(
            result.boxes,
            result.boxes.id,
        ):
            class_id = int(box.cls[0])

            class_name = get_application_class(class_id)

            if class_name is None:
                continue

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = (
                int(value)
                for value in box.xyxy[0]
            )

            tracks.append(
                Track(
                    track_id=int(track_id),
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=(x1, y1, x2, y2),
                )
            )

        return tracks