from dataclasses import dataclass
from ultralytics import YOLO

# YOLO class IDs mapped to our application classes.
YOLO_PERSON_CLASS_ID = 0

YOLO_VEHICLE_CLASS_IDS = {
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
}

PERSON_CLASS = "person"
VEHICLE_CLASS = "vehicle"



def get_application_class(class_id: int) -> str | None:
    """Map YOLO class IDs to application classes."""

    if class_id == YOLO_PERSON_CLASS_ID:
        return PERSON_CLASS

    if class_id in YOLO_VEHICLE_CLASS_IDS:
        return VEHICLE_CLASS

    return None



@dataclass
class Detection:
    """Represent a single detected object."""

    class_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[int, int, int, int]


class ObjectDetector:
    """YOLO26-based object detector."""

    def __init__(
        self,
        model_path: str = "yolo26s.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        
        self.model = YOLO(model_path)    # Load model path 
        self.confidence_threshold = confidence_threshold

    def detect(self, frame) -> list[Detection]:
        """Run object detection on a single frame."""

        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        detections: list[Detection] = []

        result = results[0]

        if result.boxes is None:
            return detections


        for box in result.boxes:
            class_id = int(box.cls[0])

            application_class = get_application_class(class_id)

            if application_class is None:
                continue        

            confidence = float(box.conf[0])
            
            x1, y1, x2, y2 = (
                int(value)
                for value in box.xyxy[0]
            )

            bounding_box = (x1, y1, x2, y2)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=application_class,
                    confidence=confidence,
                    bounding_box=bounding_box,
                )
            )

        return detections