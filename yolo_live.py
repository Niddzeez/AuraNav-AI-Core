import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8n model (n is for 'nano', the smallest version)
# This will use the yolov8n.pt file you've already downloaded.
model = YOLO(r'C:\Users\HP\OneDrive\Desktop\AuraNav_sprint\runs\detect\train\weights\best.pt')

# Open the default webcam (usually index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam feed... Press 'q' to quit.")

# Loop through the video frames
while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, stream=True)

        # ---- THE FIX IS HERE ----
        # Loop through the results generator
        for r in results:
            # The r.plot() method returns a frame with detections drawn on it
            annotated_frame = r.plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Live Detection", annotated_frame)
        # -------------------------

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the stream ends
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")