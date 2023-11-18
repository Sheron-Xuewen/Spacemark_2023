import cv2
import numpy as np
import pytesseract
import config
import re
import os
import sys



def read_config():
    with open("config.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        key, value = line.strip().split("=")
        value = value.strip("\"")
        if key in config.__dict__:
            config.__dict__[key] = value


read_config()  # 在其他代码执行前调用

# Setting Tesseract path
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD_PATH

def calculate_frame_brightness(frame):
    """Calculate the average brightness of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def extract_tilt_angle(text):
    match = re.search(r"Tilt: (\d+\.\d+|\d+)", text)
    if match:
        return float(match.group(1))
    return 0  # If no match is found, return 0

def determine_phone_position(tilt_angle):
    if config.SCREEN_UP_MIN <= tilt_angle <= config.SCREEN_UP_MAX:
        return "UP"
    else:
        return "DOWN"

# Function to get angle from direction
def get_angle_from_direction(direction_str):
    parts = direction_str.split(",")
    primary_dir = parts[0].split()[-3]
    secondary_dir = parts[0].split()[-1]

    # Check if the string can be converted to a encoding='utf-8float
    try:
        degree_offset = float(parts[1].split()[0].replace("°", ""))
    except ValueError:  # if conversion fails, set degree_offset to 0
        degree_offset = 0.0

    angle = config.DIRECTION_MAPPING[f"{primary_dir} by {secondary_dir}"] + degree_offset
    return angle


def get_angle_difference(angle1, angle2):
    diff = angle2 - angle1
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff

def determine_orientation(text, user_input_angle):
    match = re.search(r"Direction.*?from (South|North)", text)
    if match:
        angle_from_frame = get_angle_from_direction(match.group())
        angle_difference = get_angle_difference(user_input_angle, angle_from_frame)

        if config.FRONT_MIN <= angle_difference <= config.FRONT_MAX:
            return "FRONT"
        elif (config.LEFT_MIN <= angle_difference <= config.LEFT_MAX):
            return "LEFT"
        elif (config.RIGHT_MIN <= angle_difference <= config.RIGHT_MAX):
            return "RIGHT"
        elif (config.BACK_MIN_1 <= angle_difference <= config.BACK_MAX_1) or (
                config.BACK_MIN_2 <= angle_difference <= config.BACK_MAX_2):
            return "BACK"
    return config.DEFAULT_ORIENTATION

# Overlay functions
def overlay_centered(bg_img, glow_img, crosshair_img, time_of_day, grayscale=False):
    if crosshair_img is None or glow_img is None:
        return bg_img
    overlay_size = int(bg_img.shape[1] / 2)
    glow_size = int((127 / 120) * overlay_size)

    # Adjust crosshair color based on time of day
    # Define a color range for selecting pixels
    lower_bound = np.array([0, 0, 0, 255])
    upper_bound = np.array([50, 50, 50, 255])
    mask = cv2.inRange(crosshair_img, lower_bound, upper_bound)

    if time_of_day == "DAY":
        crosshair_img = modify_crosshair_color(crosshair_img, config.CROSSHAIR_COLOR_DAY)
    else:
        crosshair_img = modify_crosshair_color(crosshair_img, config.CROSSHAIR_COLOR_NIGHT)

    crosshair_img_resized = cv2.resize(crosshair_img, (overlay_size, overlay_size))
    glow_img_resized = cv2.resize(glow_img, (glow_size, glow_size))

    # Apply grayscale effect to green regions of glow_img if required
    # Apply grayscale effect only to green regions of glow_img if required
    if grayscale:
        green_mask = cv2.inRange(glow_img_resized, np.array([0, 128, 0, 0]), np.array([128, 255, 128, 255]))
        only_green = cv2.bitwise_and(glow_img_resized, glow_img_resized, mask=green_mask)
        gray_green = cv2.cvtColor(only_green, cv2.COLOR_BGRA2GRAY)
        gray_green_colored = cv2.cvtColor(gray_green, cv2.COLOR_GRAY2BGRA)
        glow_img_resized[:, :, 0:3] = glow_img_resized[:, :, 0:3] - only_green[:, :, 0:3] + gray_green_colored[:, :,
                                                                                            0:3]

    alpha_value = config.CROSSHAIR_GREEN_GLOW_DAY_ALPHA
    glow_img_resized[:, :, 3] = (alpha_value * glow_img_resized[:, :, 3]).astype(np.uint8)

    y1_glow = (bg_img.shape[0] - glow_size) // 2
    y2_glow = y1_glow + glow_size
    x1_glow = (bg_img.shape[1] - glow_size) // 2
    x2_glow = x1_glow + glow_size

    y1_crosshair = (bg_img.shape[0] - overlay_size) // 2
    y2_crosshair = y1_crosshair + overlay_size
    x1_crosshair = (bg_img.shape[1] - overlay_size) // 2
    x2_crosshair = x1_crosshair + overlay_size

    alpha_glow = glow_img_resized[:, :, 3] / 255.0
    for c in range(3):
        bg_img[y1_glow:y2_glow, x1_glow:x2_glow, c] = (
                alpha_glow * glow_img_resized[:, :, c] + (1 - alpha_glow) * bg_img[y1_glow:y2_glow, x1_glow:x2_glow,
                                                                            c]).astype(np.uint8)

    alpha_crosshair = crosshair_img_resized[:, :, 3] / 255.0
    for c in range(3):
        bg_img[y1_crosshair:y2_crosshair, x1_crosshair:x2_crosshair, c] = (
                alpha_crosshair * crosshair_img_resized[:, :, c] + (1 - alpha_crosshair) * bg_img[
                                                                                           y1_crosshair:y2_crosshair,
                                                                                           x1_crosshair:x2_crosshair,
                                                                                           c]).astype(np.uint8)

        # Now blend the grayscale processed glow_img_resized with bg_img

    return bg_img

def adjust_pixel_value(pixel, target):
    """Adjusts the pixel value based on the target color."""
    if pixel[3] == 0:
        return pixel
    return (target[0], target[1], target[2], pixel[3])

def modify_crosshair_color(crosshair, target_color):
    # Convert to BGRA for transparency handling
    crosshair = cv2.cvtColor(crosshair, cv2.COLOR_BGR2BGRA)

    # Define a threshold. Any pixel with an average value below this in all B, G, R channels will be considered "near black".
    threshold = 50

    # Calculate the average of the B, G, R channels
    avg_color = np.mean(crosshair[:, :, :3], axis=2)

    # Create a mask of pixels where the average B, G, and R values are below the threshold AND the pixel is not transparent
    mask = (avg_color < threshold) & (crosshair[:, :, 3] > 0)

    # Change color of the masked pixels
    crosshair[mask] = target_color

    return crosshair


def overlay_arrow(bg_img, arrow_img, time_of_day, arrow_left_img, arrow_right_img, arrow_turnaround_img):
    if arrow_img is None:
        return bg_img

    # Determine the type of arrow and set the vertical position ratio
    if np.array_equal(arrow_img, arrow_left_img):
        vertical_position_ratio = config.ARROW_LEFT_VERTICAL_POSITION_RATIO
        width_ratio = config.ARROW_LEFT_WIDTH_RATIO
        if time_of_day == "DAY":
            arrow_img = modify_crosshair_color(arrow_img, config.ARROW_LEFT_COLOR_DAY)
        else:
            arrow_img = modify_crosshair_color(arrow_img, config.ARROW_LEFT_COLOR_NIGHT)
    elif np.array_equal(arrow_img, arrow_right_img):
        vertical_position_ratio = config.ARROW_RIGHT_VERTICAL_POSITION_RATIO
        width_ratio = config.ARROW_RIGHT_WIDTH_RATIO
        if time_of_day == "DAY":
            arrow_img = modify_crosshair_color(arrow_img, config.ARROW_RIGHT_COLOR_DAY)
        else:
            arrow_img = modify_crosshair_color(arrow_img, config.ARROW_RIGHT_COLOR_NIGHT)
    elif np.array_equal(arrow_img, arrow_turnaround_img):
        vertical_position_ratio = config.ARROW_TURNAROUND_VERTICAL_POSITION_RATIO
        width_ratio = config.ARROW_TURNAROUND_WIDTH_RATIO
        if time_of_day == "DAY":
            arrow_img = modify_crosshair_color(arrow_img, config.ARROW_TURNAROUND_COLOR_DAY)
        else:
            arrow_img = modify_crosshair_color(arrow_img, config.ARROW_TURNAROUND_COLOR_NIGHT)

    # Resize arrow based on width ratio
    width = int(bg_img.shape[1] * width_ratio)
    aspect_ratio = arrow_img.shape[1] / arrow_img.shape[0]
    height = int(width / aspect_ratio)
    arrow_img_resized = cv2.resize(arrow_img, (width, height))

    # Position the arrow
    x1 = (bg_img.shape[1] - width) // 2
    y1 = int(bg_img.shape[0] * vertical_position_ratio) - (height // 2)
    x2 = x1 + width
    y2 = y1 + height

    alpha_arrow = arrow_img_resized[:, :, 3] / 255.0
    for c in range(3):
        bg_img[y1:y2, x1:x2, c] = (
            alpha_arrow * arrow_img_resized[:, :, c] +
            (1 - alpha_arrow) * bg_img[y1:y2, x1:x2, c]
        ).astype(np.uint8)

    return bg_img


def overlay_transparent(background_img, img_to_overlay, x, y, overlay_size=None):

    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay = cv2.resize(img_to_overlay.copy(), overlay_size)


    b, g, r, a = cv2.split(img_to_overlay)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay.shape

    # Ensure the coordinates do not go out of bounds
    y1 = max(int(y - h / 2), 0)
    y2 = min(int(y + h / 2), bg_img.shape[0])
    x1 = max(int(x - w / 2), 0)
    x2 = min(int(x + w / 2), bg_img.shape[1])

    if (y2 - y1) <= 0 or (x2 - x1) <= 0:
        print(f"Invalid ROI dimensions. y1: {y1}, y2: {y2}, x1: {x1}, x2: {x2}")
        return background_img  # Return the original image without overlaying

    roi = bg_img[y1:y2, x1:x2]

    if roi.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

    img_to_overlay_resized = cv2.resize(img_to_overlay, (roi.shape[1], roi.shape[0]))
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_resized, img_to_overlay_resized, mask=mask)

    bg_img[y1:y2, x1:x2] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


def overlay_user_image(frame, user_image):
    if user_image is None:
        # print("User image is None. Skipping overlay.")
        return frame

    # Get frame dimensions
    frame_h, frame_w, _ = frame.shape

    # Get the aspect ratio of the user image
    aspect_ratio = user_image.shape[1] / user_image.shape[0]

    # Set the maximum width and height to 1/3 of the video's width
    max_width = frame_w // 3
    max_height = frame_w // 3

    # Calculate the new width and height while preserving the aspect ratio
    new_w = max_width
    new_h = int(new_w / aspect_ratio)

    # If the new height exceeds the maximum height, adjust the width and height accordingly
    if new_h > max_height:
        new_h = max_height
        new_w = int(new_h * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(user_image, (new_w, new_h))

    # Define overlay position
    x_center = frame_w // 2
    y_center = frame_h * 2 // 14  # the vertical center of the image should be 1/14th from the top of the frame

    return overlay_transparent(frame, resized_image, x_center, y_center)

def overlay_user_text(frame, user_text):
    if user_text is None or user_text == "":
        # print("User text is empty or None. Skipping overlay.")
        return frame

    # Get frame dimensions
    frame_h, frame_w, _ = frame.shape

    # Define font, size, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9  # Adjusted value to reduce the font size
    font_thickness = 2
    color = (0, 0, 0)  # Black text
    background_color = (255, 255, 255)  # White background

    # Get text size
    text_size = cv2.getTextSize(user_text, font, font_scale, font_thickness)[0]

    # Adjust text position
    text_x = (frame_w - text_size[0]) // 2
    text_y = (frame_h * 4 // 14) + text_size[1] + 10  # Adjusted position

    # Draw a background rectangle for the text
    rectangle_bgr = (255, 255, 255)  # White rectangle
    (text_width, text_height), _ = cv2.getTextSize(user_text, font, font_scale, font_thickness)
    # Set the rectangle background to white
    frame[text_y - text_height - 10:text_y + 10, text_x - 10:text_x + text_width + 10] = rectangle_bgr

    # Overlay text on the frame
    cv2.putText(frame, user_text, (text_x, text_y), font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    return frame


# Main function
def main_overlay_with_glow():

    grayscale_effect = False
    cap = cv2.VideoCapture(config.VIDEO_PATH)

    # Load user's custom text
    user_text = config.USER_TEXT

    # Load user's custom image
    user_image = None
    if os.path.exists(config.USER_IMAGE_PATH) and (
            config.USER_IMAGE_PATH.endswith('.jpg') or config.USER_IMAGE_PATH.endswith('.png')):
        user_image_temp = cv2.imread(config.USER_IMAGE_PATH)
        if user_image_temp is not None:
            user_image = cv2.cvtColor(user_image_temp, cv2.COLOR_BGR2BGRA)
        else:
            print(f"Failed to load user image from: {config.USER_IMAGE_PATH}")
    else:
        print(f"User image path is either incorrect or not a supported format: {config.USER_IMAGE_PATH}")

    if not cap.isOpened():
        print("Error opening video file!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
    out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    crosshair_img = cv2.imread(config.CROSSHAIR_PATH, -1)
    orange_glow_img = cv2.imread(config.CROSSHAIR_ORANGE_GLOW_PATH, -1)
    green_glow_img = cv2.imread(config.CROSSHAIR_GREEN_GLOW_PATH, -1)
    arrow_left_img = cv2.imread(config.ARROW_LEFT_PATH, -1)
    arrow_right_img = cv2.imread(config.ARROW_RIGHT_PATH, -1)
    arrow_turnaround_img = cv2.imread(config.ARROW_TURNAROUND_PATH, -1)


    user_input_string = config.USER_DIRECTION_INPUT
    user_input_angle = get_angle_from_direction(user_input_string)

    frame_count = 0

    direction_recognized = True

    time_of_day = "DAY"  # Default value
    current_arrow = None

    # Initialize for the first frame
    ret, frame = cap.read()
    if ret:
        tesseract_output = pytesseract.image_to_string(frame)
        print(f"Frame {frame_count} OCR Output: {tesseract_output}")

        # Extract tilt angle
        tilt_angle = extract_tilt_angle(tesseract_output)
        print(f"Frame {frame_count} Tilt Angle: {tilt_angle}")

        phone_position = determine_phone_position(tilt_angle)
        print(f"Frame {frame_count} Phone Position: {phone_position}")

        # Determine day or night
        brightness = calculate_frame_brightness(frame)
        if brightness <= config.BRIGHTNESS_THRESHOLD:
            time_of_day = "NIGHT"
        else:
            time_of_day = "DAY"
        print(f"Frame {frame_count} Time of Day: {time_of_day}")

        orientation = determine_orientation(tesseract_output, user_input_angle)
        print(f"Frame {frame_count} Orientation: {orientation}")

        if not direction_recognized:
            user_image = None
            user_text = None

        if orientation == config.DEFAULT_ORIENTATION and "Direction" not in tesseract_output:
            direction_recognized = False
        else:
            direction_recognized = True

        is_default_orientation = (orientation == config.DEFAULT_ORIENTATION)
        if orientation == "FRONT":
            current_overlay = green_glow_img
        else:
            current_overlay = orange_glow_img


        # Overlay the crosshair and glow using the overlay_centered function
        overlaid_frame = overlay_centered(frame.copy(), current_overlay, crosshair_img, time_of_day, grayscale=False)

        out.write(overlaid_frame)
        frame_count += 1

        current_arrow = None
        if orientation == "LEFT":
            current_arrow = arrow_left_img
        elif orientation == "RIGHT":
            current_arrow = arrow_right_img
        elif orientation == "BACK":
            current_arrow = arrow_turnaround_img

    # Continue for the rest of the frames
    while cap.isOpened():
        ret, frame = cap.read()

        # Check if frame is None
        if frame is None:
            break

        if frame_count % fps == 0:
            if not ret:
                break

        frame_count += 1

        # Apply the overlay to the frame with or without grayscale as required
        # First, overlay the crosshair and glow on the frame.
        overlaid_frame = overlay_centered(frame.copy(), current_overlay, crosshair_img, time_of_day,
                                          grayscale=grayscale_effect)

        # Then, overlay the arrow if there is one.
        if current_arrow is not None:
            overlaid_frame = overlay_arrow(overlaid_frame, current_arrow, time_of_day, arrow_left_img,
                                           arrow_right_img, arrow_turnaround_img)

        # Only overlay the user image and text if orientation is FRONT
        if orientation == "FRONT":
            overlaid_frame = overlay_user_image(overlaid_frame, user_image)
            overlaid_frame = overlay_user_text(overlaid_frame, user_text)

        # Write the frame to the output
        out.write(overlaid_frame)

        if frame_count % fps == 0:
            # OCR processing for the first frame every second based on fps
            tesseract_output = pytesseract.image_to_string(frame)

            print(f"Frame {frame_count} OCR Output: {tesseract_output}")

            # Extract tilt angle
            tilt_angle = extract_tilt_angle(tesseract_output)

            phone_position = determine_phone_position(tilt_angle)

            # Determine day or night
            brightness = calculate_frame_brightness(frame)
            if brightness <= config.BRIGHTNESS_THRESHOLD:
                time_of_day = "NIGHT"
            else:
                time_of_day = "DAY"

            orientation = determine_orientation(tesseract_output, user_input_angle)

            print(f"Frame {frame_count} Tilt Angle: {tilt_angle}")
            print(f"Frame {frame_count} Phone Position: {phone_position}")
            print(f"Frame {frame_count} Time of Day: {time_of_day}")
            print(f"Frame {frame_count} Orientation: {orientation}")

            if not direction_recognized:
                user_image = None
                user_text = None

            if "Direction" not in tesseract_output:
                direction_recognized = False
            else:
                direction_recognized = True

            # Determine the correct arrow to overlay based on orientation
            if orientation == "LEFT":
                current_arrow = arrow_right_img
            elif orientation == "RIGHT":
                current_arrow = arrow_left_img
            elif orientation == "BACK":
                current_arrow = arrow_turnaround_img

            if phone_position == "DOWN":
                if orientation == "LEFT" and config.ARROW_LEFT_HIDE_ON_SCREEN_DOWN:
                    current_arrow = None
                elif orientation == "RIGHT" and config.ARROW_RIGHT_HIDE_ON_SCREEN_DOWN:
                    current_arrow = None
                elif orientation == "BACK" and config.ARROW_TURNAROUND_HIDE_ON_SCREEN_DOWN:
                    current_arrow = None

            # Decide which overlay to use based on orientation and display conditions
            if not direction_recognized or orientation == config.CROSSHAIR_GREEN_GLOW_DISPLAY_CONDITION:
                current_overlay = green_glow_img
            else:
                current_overlay = orange_glow_img

            if not direction_recognized:
                user_image = None
                user_text = None

            # Overlay the crosshair and glow using the overlay_centered function
            overlaid_frame = overlay_centered(frame.copy(), current_overlay, crosshair_img, time_of_day)

            # If the orientation is "FRONT" and direction is recognized, overlay the user-provided image and text
            if orientation == "FRONT" and direction_recognized:
                overlaid_frame = overlay_user_image(overlaid_frame, user_image)
                overlaid_frame = overlay_user_text(overlaid_frame, user_text)

            # Apply grayscale effect for default orientation only
            grayscale_effect = is_default_orientation

            # Hide overlay if phone position is DOWN and corresponding setting is True
            if phone_position == "DOWN":
                if current_overlay is green_glow_img and config.CROSSHAIR_GREEN_GLOW_HIDE_ON_SCREEN_DOWN:
                    current_overlay = None
                elif current_overlay is orange_glow_img and config.CROSSHAIR_ORANGE_GLOW_HIDE_ON_SCREEN_DOWN:
                    current_overlay = None
                if config.CROSSHAIR_HIDE_ON_SCREEN_DOWN:
                    crosshair_img = None


    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_overlay_with_glow()