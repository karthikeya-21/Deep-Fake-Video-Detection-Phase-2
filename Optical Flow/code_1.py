import cv2
import numpy as np
import os
# Replace 'your_video.mp4' with the path to your video file
video_path = '000.mp4'

# Create VideoCapture objects
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame1 = cap.read()

# Create a directory to save the optical flow images
output_dir = 'optical_flow_images3'
os.makedirs(output_dir, exist_ok=True)

# Parameters for Dense Optical Flow
params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# Initialize previous frame
prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Counter for image file names
image_counter = 0

while True:
    # Read the next frame
    ret, frame2 = cap.read()

    if not ret:
        break

    # Convert the current frame to grayscale
    current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, **params)

    # Calculate magnitude and angle of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an HSV image to visualize the optical flow
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for display
    optical_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the optical flow image
    image_counter += 1
    image_filename = f'{output_dir}/optical_flow_{image_counter:04d}.png'
    cv2.imwrite(image_filename, optical_flow)

    # Display the optical flow image
    cv2.imshow('Optical Flow', optical_flow)

    # Update the previous frame
    prev_frame = current_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close the window
cap.release()
cv2.destroyAllWindows()
