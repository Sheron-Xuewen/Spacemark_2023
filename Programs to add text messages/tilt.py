import cv2
import numpy as np

# Input and output file paths
input_video_path = 'E:\\AR_Wayfinding_Project\\assets\\original\\UpwardFacing_Original.mp4'
output_video_path = 'E:\\AR_Wayfinding_Project\\assets\\original\\UpwardFacing_Original_Processed.mp4'

# Open the video and get its properties
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize fixed direction info
direction_info = "Direction: South by West, 42.94 degrees from South"

# Calculate tilt change per frame
start_tilt_angle = 80  # Starting from 80 degrees
end_tilt_angle = 150  # Ending at 180 degrees
tilt_change_per_frame = (end_tilt_angle - start_tilt_angle) / frames_count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Determine tilt description
    tilt_info = f"Tilt: {start_tilt_angle:.2f} degrees"
    (w_dir, h_dir), _ = cv2.getTextSize(direction_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    (w_tilt, h_tilt), _ = cv2.getTextSize(tilt_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # Calculate the total height needed for both texts and their backgrounds
    total_height = h_dir + h_tilt + 20

    # Add a semi-transparent rectangle behind both texts
    cv2.rectangle(frame, (5, 5), (5 + max(w_dir, w_tilt) + 20, 5 + total_height), (255, 255, 255), -1)

    # Adjust the vertical position of the texts so they're centered in the rectangle
    text_y_dir = 5 + h_dir + (h_dir // 4)
    text_y_tilt = text_y_dir + h_tilt + 10

    # Add the direction and tilt texts
    cv2.putText(frame, direction_info, (10, text_y_dir), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, tilt_info, (10, text_y_tilt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Increment the tilt angle for next frame
    start_tilt_angle += tilt_change_per_frame

    # Save the processed frame
    out.write(frame)

# Close video files
cap.release()
out.release()

print("Video processing completed.")
