import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Load the MiDaS model and processor from Hugging Face
# This will download the model on the first run
processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam feed... Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Prepare the image for the model
        inputs = processor(images=frame, return_tensors="pt")

        # Run the model to get the depth estimation
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate the prediction to the original image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Format the output for display
        output_depth = prediction.cpu().numpy()
        formatted_depth = (output_depth * 255 / np.max(output_depth)).astype("uint8")
        depth_colormap = cv2.applyColorMap(formatted_depth, cv2.COLORMAP_MAGMA)


        # Combine the original frame and the depth map to display side-by-side
        combined_view = np.hstack((frame, depth_colormap))

        cv2.imshow("Live Feed | MiDaS Depth Map", combined_view)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")