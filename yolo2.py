import cv2
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION ---
# 1. PATH TO YOUR CUSTOM MODEL
MODEL_PATH = r'C:\Users\HP\OneDrive\Desktop\AuraNav_sprint\runs\detect\train\weights\best.pt'

# 2. PATH TO THE VIDEO FILE YOU WANT TO TEST
VIDEO_FILE_PATH = r"F:\Videos\Hostel_Indoors\VID_20251011_225400.mp4"

# 3. SET THE DISPLAY HEIGHT (in pixels). 720 is a good standard size.
DISPLAY_HEIGHT = 720
# ---------------------


# Load your custom-trained model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open the video file
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

# Check if the video opened correctly
if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_FILE_PATH}")
    exit()

print("Processing video file... Press 'q' to quit.")

# Loop through the video frames
while True:
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, stream=True)

        # Loop through the results generator
        for r in results:
            # The r.plot() method returns a frame with detections drawn on it
            annotated_frame = r.plot()

            # ---- NEW RESIZING CODE STARTS HERE ----
            # Get the original dimensions
            h, w, _ = annotated_frame.shape
            
            # Calculate the new width to maintain the aspect ratio
            aspect_ratio = w / h
            new_width = int(DISPLAY_HEIGHT * aspect_ratio)
            
            # Resize the frame
            resized_frame = cv2.resize(annotated_frame, (new_width, DISPLAY_HEIGHT))
            # ---- NEW RESIZING CODE ENDS HERE ----

            # Display the RESIZED frame
            cv2.imshow("Custom Model - Video Test", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the video has ended
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
print("Video processing finished.")