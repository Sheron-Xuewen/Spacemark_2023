import cv2
import numpy as np
from sklearn.cluster import KMeans


def load_image(image_path):
    """Load an image from a given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not load image!")
    return image


def detect_endpoints(image):
    red_centroid = None
    endpoint_centroids = []
    endpoint_colors = []

    # Convert to grayscale and apply adaptive thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (to distinguish between starting point and endpoints)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # The largest contour is the starting point
    M = cv2.moments(sorted_contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    red_centroid = (cY, cX)

    # Remaining contours are endpoints
    for contour in sorted_contours[1:]:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        color = image[cY, cX]
        endpoint_centroids.append((cY, cX))
        endpoint_colors.append(tuple(color))

    return red_centroid, endpoint_centroids, endpoint_colors


def compute_direction(angle_from_east):
    """Compute the cardinal direction based on the angle from East."""
    if 0 <= angle_from_east < 90:
        return f"East by South, {angle_from_east:.2f}° from East"
    elif 90 <= angle_from_east < 180:
        return f"South by West, {angle_from_east - 90:.2f}° from South"
    elif 180 <= angle_from_east < 270:
        return f"West by North, {angle_from_east - 180:.2f}° from West"
    elif 270 <= angle_from_east < 360:
        return f"North by East, {angle_from_east - 270:.2f}° from North"


# ... [rest of the imports and function definitions]

def compute_distances_and_directions(red_centroid, endpoint_centroids):
    distances_and_directions = []
    for endpoint in endpoint_centroids:
        dx = endpoint[1] - red_centroid[1]
        dy = endpoint[0] - red_centroid[0]

        angle_with_horizontal = np.degrees(np.arctan2(dy, dx))
        adjusted_angle = (angle_with_horizontal + 360) % 360
        pixel_distance = np.sqrt(dx ** 2 + dy ** 2)

        distances_and_directions.append((pixel_distance, adjusted_angle))

    # Sort based on distance
    sorted_distances_and_directions, sorted_endpoints = zip(*sorted(zip(distances_and_directions, endpoint_centroids)))

    return sorted_endpoints, sorted_distances_and_directions


if __name__ == "__main__":
    image_path = "E:\\AR_Wayfinding_Project\\assets\\Multitarget.png"
    image = load_image(image_path)
    pixel_to_meter_ratio = 0.1

    red_centroid, endpoint_centroids, endpoint_colors = detect_endpoints(image)
    sorted_endpoints, sorted_distances_and_directions = compute_distances_and_directions(red_centroid,
                                                                                        endpoint_centroids)

    # Mark the starting point (red)
    cv2.circle(image, (int(red_centroid[1]), int(red_centroid[0])), 10, (0, 0, 255), -1)
    print("Starting Point:")
    print(f"Position: {red_centroid}")

    for i, (endpoint, (distance, direction)) in enumerate(zip(sorted_endpoints, sorted_distances_and_directions)):
        # Displaying the results on the image
        color = image[endpoint[0], endpoint[1]]
        cv2.circle(image, (int(endpoint[1]), int(endpoint[0])), 10, tuple(map(int, color)), -1)
        cv2.line(image, (int(red_centroid[1]), int(red_centroid[0])), (int(endpoint[1]), int(endpoint[0])),
                 tuple(map(int, color)), 2)

        # Printing distances and directions
        real_world_distance = distance * pixel_to_meter_ratio
        print(f"Endpoint {i + 1}:")
        print(f"Real-world distance from start: {real_world_distance} meters")
        print(f"Direction from start: {compute_direction(direction)}")

    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
