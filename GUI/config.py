# config.py
import cv2

# ============ Paths ============
CROSSHAIR_PATH = "E:\\AR_Wayfinding_Project\\assets\\gallery\\crosshair.png"
CROSSHAIR_ORANGE_GLOW_PATH = "E:\\AR_Wayfinding_Project\\assets\\gallery\\crosshair_orange_glow.png"
CROSSHAIR_GREEN_GLOW_PATH = "E:\\AR_Wayfinding_Project\\assets\\gallery\\crosshair_green_glow.png"
ARROW_LEFT_PATH = "E:\\AR_Wayfinding_Project\\assets\\gallery\\arrow_left.png"
ARROW_RIGHT_PATH = "E:\\AR_Wayfinding_Project\\assets\\gallery\\arrow_right.png"
ARROW_TURNAROUND_PATH = "E:\\AR_Wayfinding_Project\\assets\\gallery\\arrow_turnaround.png"
UPRIGHT_REMINDER_PATH = "E:\\AR_Wayfinding_Project\\assets\\gallery\\upright_reminder.png"

USER_TEXT = ""
USER_DIRECTION_INPUT = ""
VIDEO_PATH = ""
USER_IMAGE_PATH = ""
OUTPUT_VIDEO_PATH = ""

# Function to write configurations to a file
def write_config():
    with open("config.txt", "w") as f:
        f.write(f"USER_TEXT={USER_TEXT}\n")
        f.write(f"USER_DIRECTION_INPUT={USER_DIRECTION_INPUT}\n")
        f.write(f"VIDEO_PATH={VIDEO_PATH}\n")
        f.write(f"USER_IMAGE_PATH={USER_IMAGE_PATH}\n")
        f.write(f"OUTPUT_VIDEO_PATH={OUTPUT_VIDEO_PATH}\n")

# ============ Tesseract OCR Configuration ============
TESSERACT_CMD_PATH = 'E:\\Tesseract\\tesseract.exe'

# ============ Day/Night Brightness Threshold ============
BRIGHTNESS_THRESHOLD = 100  # Determine day or night based on frame brightness

# ============ Directional Mapping ============
# Mapping of directions to their respective angles
DIRECTION_MAPPING = {
    "North by East": 45,
    "East": 90,
    "South by East": 135,
    "South": 180,
    "South by West": 225,
    "West": 270,
    "North by West": 315
}

# ============ Orientation Thresholds ============
# Determine which direction the user is facing
FRONT_MIN = -15
FRONT_MAX = 15
BACK_MIN_1 = 165
BACK_MAX_1 = 180
BACK_MIN_2 = -180
BACK_MAX_2 = -165
LEFT_MIN = -165
LEFT_MAX = -15
RIGHT_MIN = 15
RIGHT_MAX = 165

# ============= Device Tilt Information =============
# Determine the tilt of the device based on detected angle
SCREEN_UP_MIN = 0  # Minimum angle for screen facing up
SCREEN_UP_MAX = 120  # Maximum angle for screen facing up

# ============ Default Values ============
# Default orientation and tilt values if they can't be determined
DEFAULT_ORIENTATION = "FRONT"
GRAY_OUT_OVERLAY = True
GRAYSCALE_EFFECT: False
DEFAULT_TILT = "SCREEN_UP"

# ============ Graphics Display Settings ============

# ------------ CROSSHAIR Settings ------------
CROSSHAIR_POSITION = "center"  # Always load in the center of the video
CROSSHAIR_DIMENSION_RATIO = 0.5  # Side length is half the width of the video

CROSSHAIR_COLOR_DAY = (0, 0, 0, 255)  # Display color during daytime
CROSSHAIR_COLOR_NIGHT = (255,255,255,255)  # Display color during nighttime

# CROSSHAIR Screen Facing Down Behavior
CROSSHAIR_HIDE_ON_SCREEN_DOWN = True  # If the device screen is detected to be facing down, the crosshair will not display


# ------------ CROSSHAIR ORANGE GLOW Settings ------------
CROSSHAIR_ORANGE_GLOW_POSITION = "center" # Always load in the center of the video
CROSSHAIR_ORANGE_GLOW_DIMENSION_RATIO = 0.5 * (127 / 120) # Glow dimension slightly larger than crosshair

CROSSHAIR_ORANGE_GLOW_DAY_ALPHA = 0.3  # 40% transparency during daytime
CROSSHAIR_ORANGE_GLOW_NIGHT_ALPHA = 0.5  # 60% transparency during nighttime

CROSSHAIR_ORANGE_GLOW_DISPLAY_CONDITION = "not_front"  # Only display when user is NOT facing forward

# CROSSHAIR_ORANGE_GLOW Screen Facing Down Behavior
CROSSHAIR_ORANGE_GLOW_HIDE_ON_SCREEN_DOWN = True  # If the device screen is detected to be facing down, the crosshair_orange_glow will not display

# ------------ CROSSHAIR GREEN GLOW Settings ------------
CROSSHAIR_GREEN_GLOW_POSITION = "center" # Always load in the center of the video
CROSSHAIR_GREEN_GLOW_DIMENSION_RATIO = 0.5 * (13 / 12) # Glow dimension slightly larger than crosshair

CROSSHAIR_GREEN_GLOW_DAY_ALPHA = 0.5  # 60% transparency during daytime
CROSSHAIR_GREEN_GLOW_NIGHT_ALPHA = 0.6  # 80% transparency during nighttime

CROSSHAIR_GREEN_GLOW_DISPLAY_CONDITION = "front"  # Only display when user is facing forward

# CROSSHAIR_GREEN_GLOW Screen Facing Down Behavior
CROSSHAIR_GREEN_GLOW_HIDE_ON_SCREEN_DOWN = True  # If the device screen is detected to be facing down, the crosshair_green_glow will not display

# ------------ ARROW LEFT Settings ------------
ARROW_LEFT_ASPECT_RATIO = "keep"  # Keep the original aspect ratio of the graphic
ARROW_LEFT_WIDTH_RATIO = 0.5  # Arrow width is half of the video width
ARROW_LEFT_VERTICAL_POSITION_RATIO = 11/14  # Position the center of the arrow so it's distance from the bottom is 3/14 of video height

ARROW_LEFT_COLOR_DAY = (0,0,0,255)  # Display color during daytime
ARROW_LEFT_COLOR_NIGHT = (255,255,255,255)  # Display color during nighttime

ARROW_LEFT_DISPLAY_CONDITION = "right"  # Only display when user is facing right

# ARROW_LEFT Screen Facing Down Behavior
ARROW_LEFT_HIDE_ON_SCREEN_DOWN = True  # If the device screen is detected to be facing down, the arrow_left will not display

# ------------ ARROW RIGHT Settings ------------
ARROW_RIGHT_ASPECT_RATIO = "keep"  # Keep the original aspect ratio of the graphic
ARROW_RIGHT_WIDTH_RATIO = 0.5  # Arrow width is half of the video width
ARROW_RIGHT_VERTICAL_POSITION_RATIO = 11/14  # Position the center of the arrow so it's distance from the bottom is 3/14 of video height

ARROW_RIGHT_COLOR_DAY = (0,0,0,255)  # Display color during daytime
ARROW_RIGHT_COLOR_NIGHT = (255,255,255,255)  # Display color during nighttime

ARROW_RIGHT_DISPLAY_CONDITION = "left"  # Only display when user is facing left

# ARROW_RIGHT Screen Facing Down Behavior
ARROW_RIGHT_HIDE_ON_SCREEN_DOWN = True  # If the device screen is detected to be facing down, the arrow_right will not display

# ------------ ARROW TURNAROUND Settings ------------
ARROW_TURNAROUND_ASPECT_RATIO = "keep"  # Keep the original aspect ratio of the graphic
ARROW_TURNAROUND_WIDTH_RATIO = 0.5  # Arrow width is half of the video width
ARROW_TURNAROUND_VERTICAL_POSITION_RATIO = 11/14  # Position the center of the arrow so it's distance from the bottom is 3/14 of video height

ARROW_TURNAROUND_COLOR_DAY = (0,0,0,255)  # Display color during daytime
ARROW_TURNAROUND_COLOR_NIGHT = (255,255,255,255)  # Display color during nighttime

ARROW_TURNAROUND_DISPLAY_CONDITION = "back"  # Only display when user is facing back

# ARROW_TURNAROUND Screen Facing Down Behavior
ARROW_TURNAROUND_HIDE_ON_SCREEN_DOWN = True  # If the device screen is detected to be facing down, the arrow_turnaround will not display

# ------------ USER LABEL Settings ------------
# Image settings
IMAGE_MAX_WIDTH_RATIO = 0.5  # Maximum width of the image as a fraction of video width
IMAGE_MAX_HEIGHT_RATIO = 0.5  # Maximum height of the image also as a fraction of video width (assuming both width and height are the same based on your description)
IMAGE_VERTICAL_OFFSET_RATIO = 1/14  # Vertical offset from the top of the video to the centroid of the image

# Text settings
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX  # Using OpenCV's font
TEXT_FONT_SIZE = 0.7  # Previously used font size, can be adjusted later
TEXT_FONT_COLOR = (255, 255, 255)  # White color for the text
TEXT_BACKGROUND_COLOR = (0, 0, 0)  # Black color for text background
TEXT_PADDING = 5  # Padding around the text for its background
TEXT_POSITION_BELOW_IMAGE = 10  # Vertical offset between the image and the text

# ------------ UPRIGHT REMINDER Settings ------------
UPRIGHT_REMINDER_ASPECT_RATIO = "keep"  # Keep the original aspect ratio of the graphic
UPRIGHT_REMINDER_WIDTH_RATIO = 0.5  # Reminder width is half of the video width
UPRIGHT_REMINDER_VERTICAL_POSITION_RATIO = 1/2  # Position the center of the reminder at the center of the video

UPRIGHT_REMINDER_HIDE_ON_SCREEN_DOWN = False

