import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image!")
        return None
    return image

def detect_red_and_blue_centroids(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    red_centroid = np.array(np.where(red_mask)).T.mean(axis=0)
    blue_centroid = np.array(np.where(blue_mask)).T.mean(axis=0)

    return red_centroid, blue_centroid

def compute_direction(angle_from_east):
    if 0 <= angle_from_east < 90:
        return f"East by South, {angle_from_east:.2f}° from East"
    elif 90 <= angle_from_east < 180:
        return f"South by West, {angle_from_east - 90:.2f}° from South"
    elif 180 <= angle_from_east < 270:
        return f"West by North, {angle_from_east - 180:.2f}° from West"
    elif 270 <= angle_from_east < 360:
        return f"North by East, {angle_from_east - 270:.2f}° from North"

if __name__ == "__main__":
    image_path = "E:\\AR_Wayfinding_Project\\assets\\Wayfinding_Schematic.png"
    image = load_image(image_path)

    if image is not None:
        red_centroid, blue_centroid = detect_red_and_blue_centroids(image)
        pixel_to_meter_ratio = 0.1  # 10 pixels = 1 meter

        dx = blue_centroid[1] - red_centroid[1]
        dy = blue_centroid[0] - red_centroid[0]

        angle_with_horizontal = np.degrees(np.arctan2(dy, dx))
        adjusted_angle = (angle_with_horizontal + 360) % 360  # Ensure angle is positive

        pixel_distance = np.sqrt(dx ** 2 + dy ** 2)
        real_world_distance = pixel_distance * pixel_to_meter_ratio

        print(f"Direction from start to end point: {compute_direction(adjusted_angle)}")

