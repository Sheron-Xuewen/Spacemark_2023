import cv2
import numpy as np

# Input and output file paths
input_video_path = 'E:\\AR_Wayfinding_Project\\demon\\original\\turn right.mp4'
output_video_path = 'E:\\AR_Wayfinding_Project\\demon\\original\\turn right_Processed.mp4'

# Open the video and get its properties
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize variables
frames_count = 29 * 3  # 3 seconds, 29 frames per second
half_frames = frames_count // 2

# For the first half, from -43 to -90 (or South by West, 43 to South by West, 90)
start_angle_1 = -43.0
end_angle_1 = -90.0
angle_change_per_frame_1 = (end_angle_1 - start_angle_1) / half_frames

# Initialize variables for the second half
start_angle_2 = 90.0  # Start at North by West, 90.00 degrees from North but displayed as West
end_angle_2 = 47.0  # End at North by West, 47.00 degrees from North
angle_change_per_frame_2 = (end_angle_2 - start_angle_2) / half_frames

# Initialize current_angle for the first half
current_angle = start_angle_1

# Initialize tilt variables for the first and second half
start_tilt_1 = 80.0  # Starting from 80
end_tilt_1 = 90.0  # Ending at 90
tilt_change_per_frame_1 = (end_tilt_1 - start_tilt_1) / half_frames

start_tilt_2 = 90.0  # Starting from 90
end_tilt_2 = 80.0  # Ending at 80
tilt_change_per_frame_2 = (end_tilt_2 - start_tilt_2) / half_frames

# Initialize current_tilt for the first half
current_tilt = start_tilt_1

frame_num = 0  # Initialize frame number

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num < half_frames:
        # First half logic
        if current_angle == -90.0:
            direction_info = "Direction: West"
        else:
            direction_info = f"Direction: South by West, {-current_angle:.2f} degrees from South"

        current_angle += angle_change_per_frame_1
        current_tilt += tilt_change_per_frame_1
    else:
        # Re-initialize for the second half if just passed the first half
        if frame_num == half_frames:
            current_angle = start_angle_2
            current_tilt = start_tilt_2
        # Second half logic
        if current_angle == 90.0:
            direction_info = "Direction: West"
        else:
            direction_info = f"Direction: North by West, {current_angle:.2f} degrees from North"

        current_angle += angle_change_per_frame_2
        current_tilt += tilt_change_per_frame_2

    (w_dir, h_dir), _ = cv2.getTextSize(direction_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # Determine tilt description
    tilt_info = f"Tilt: {current_tilt:.2f} degrees"
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

    # Update the frame_num
    frame_num += 1

    # Save the processed frame
    out.write(frame)

# Close video files
cap.release()
out.release()

print("Video processing completed.")
