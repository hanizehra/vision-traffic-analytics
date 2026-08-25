import cv2


# predefined area used to display the video.
# Tvideo keeps its original aspect ratio inside this area.
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 600

VIDEO_PATH = "data/videos/cars/c1_one_way_road.mp4"

# stores the two points coordinates(x,y).
selected_points = []


def calculate_display_size(frame_width, frame_height):
    """Calculate the proportional display size of the video."""

    width_scale = DISPLAY_WIDTH / frame_width
    height_scale = DISPLAY_HEIGHT / frame_height

    # Use the smaller scale so the entire frame fits without cropping.
    scale_factor = min(width_scale, height_scale)

    display_width = int(frame_width * scale_factor)
    display_height = int(frame_height * scale_factor)

    return display_width, display_height, scale_factor


def convert_to_original_coordinates(
    point,
    scale_factor,
    offset_x,
    offset_y,
):
    """Convert display coordinates back to original frame coordinates."""

    display_x, display_y = point

    # Remove the empty space around the resized video.
    video_display_x = display_x - offset_x
    video_display_y = display_y - offset_y

    # Reverse the scaling to get the original frame coordinates.
    original_x = int(video_display_x / scale_factor)
    original_y = int(video_display_y / scale_factor)

    return original_x, original_y


def handle_mouse_event(event, x, y, flags, parameters):
    """Store a point when the user clicks inside the window."""

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Only two points are needed to define the line.
    if len(selected_points) >= 2:
        return

    selected_points.append((x, y))

    print(f"Display coordinate: ({x}, {y})")


# Open the selected video.
video_capture = cv2.VideoCapture(VIDEO_PATH)

if not video_capture.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


# Read one frame because we only need a frame for line selection.
success, frame = video_capture.read()

if not success:
    raise RuntimeError("Could not read the first frame from the video.")


# Get the original frame dimensions.
frame_height, frame_width = frame.shape[:2]

print(f"Original frame resolution: {frame_width}x{frame_height}")


# Calculate the proportional size used for displaying the frame.
display_width, display_height, scale_factor = calculate_display_size(
    frame_width,
    frame_height,
)

print(f"Display video resolution: {display_width}x{display_height}")


# Resize the frame while keeping its original aspect ratio.
display_frame = cv2.resize(
    frame,
    (display_width, display_height),
)


# Calculate the empty space around the resized video.
offset_x = (DISPLAY_WIDTH - display_width) // 2
offset_y = (DISPLAY_HEIGHT - display_height) // 2


# Create the fixed-size canvas used by the OpenCV window.
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

    # Create a fresh copy so drawings do not modify the original canvas.
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

    # Show the line after both points have been selected.
    elif len(selected_points) == 2:

        cv2.line(
            display_image,
            selected_points[0],
            selected_points[1],
            (0, 255, 0),
            2,
        )

    # Display the current frame.
    cv2.imshow(
        "LineSelector",
        display_image,
    )

    # Process keyboard and mouse events.
    key = cv2.waitKey(1) & 0xFF

    # ESC → exit the utility.
    if key == 27:
        break

    # R → remove the selected points and start again.
    if key == ord("r"):
        selected_points.clear()
        print("Points reset.")

    # ENTER → confirm the line when two points are selected.
    if key == 13 and len(selected_points) == 2:
        break


# Convert the selected points to the original video coordinates.
if len(selected_points) == 2:

    original_point_1 = convert_to_original_coordinates(
        selected_points[0],
        scale_factor,
        offset_x,
        offset_y,
    )

    original_point_2 = convert_to_original_coordinates(
        selected_points[1],
        scale_factor,
        offset_x,
        offset_y,
    )

    print(f"Original coordinate 1: {original_point_1}")
    print(f"Original coordinate 2: {original_point_2}")


# Release resources and close the OpenCV window.
video_capture.release()
cv2.destroyAllWindows()