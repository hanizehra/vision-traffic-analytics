import cv2


DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 600


def prepare_display_frame(
    frame,
) -> tuple:
    """Resize a frame proportionally and place it on a fixed canvas."""

    frame_height, frame_width = frame.shape[:2]

    width_scale = DISPLAY_WIDTH / frame_width
    height_scale = DISPLAY_HEIGHT / frame_height

    scale_factor = min(
        width_scale,
        height_scale,
    )

    display_width = int(frame_width * scale_factor)
    display_height = int(frame_height * scale_factor)

    display_frame = cv2.resize(
        frame,
        (display_width, display_height),
    )

    offset_x = (DISPLAY_WIDTH - display_width) // 2
    offset_y = (DISPLAY_HEIGHT - display_height) // 2

    display_canvas = cv2.copyMakeBorder(
        display_frame,
        offset_y,
        DISPLAY_HEIGHT - display_height - offset_y,
        offset_x,
        DISPLAY_WIDTH - display_width - offset_x,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return (
        display_canvas,
        scale_factor,
        offset_x,
        offset_y,
    )

def convert_to_display_coordinates(
    point: tuple[int, int],
    scale_factor: float,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int]:
    """Convert original frame coordinates to display coordinates."""

    original_x, original_y = point

    display_x = int(
        original_x * scale_factor + offset_x
    )

    display_y = int(
        original_y * scale_factor + offset_y
    )

    return display_x, display_y