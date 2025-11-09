import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import win32com.client as wincl
import time
import threading
from queue import Queue
import pythoncom

# --- 1. CONFIGURATION ---
YOLO_MODEL_PATH = r'C:\Users\HP\OneDrive\Desktop\AuraNav_sprint\runs\detect\train\weights\best.pt'
VIDEO_FILE_PATH = r"F:\Videos\Hostel_Indoors\VID_20251011_225400.mp4"
DISPLAY_HEIGHT = 720
ALERT_COOLDOWN = 5 # seconds
# ---------------------

# --- TTS Worker Function (runs in the background) ---
def tts_worker(q):
    while True:
         # Initialize the COM library for this thread
        pythoncom.CoInitialize()
        # Create a single, long-lived engine instance
        speak = wincl.Dispatch("SAPI.SpVoice")

        while True:
            text = q.get()
            if text is None:
                break
            speak.Speak(text)

            q.task_done()
# --- 2. INITIALIZATION ---
print("Loading models... This may take a moment.")
tts_queue = Queue()
last_alert_time = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Start the TTS worker thread
tts_thread = threading.Thread(target=tts_worker, args=(tts_queue,))
tts_thread.daemon = True # Allows main program to exit even if thread is running
tts_thread.start()
print("TTS worker thread started.")

# Load AI models
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to(device)
    print("YOLOv8 model loaded successfully.")
    midas_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    midas_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
    print("MiDaS model loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Open video file
cap = cv2.VideoCapture(VIDEO_FILE_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_FILE_PATH}")
    exit()

print("\nStarting video processing... Press 'q' to quit.")
# --- 3. MAIN PROCESSING LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        print("Video has ended.")
        break

    # Run Inference
    yolo_results = yolo_model(frame, verbose=False) # verbose=False cleans up terminal output
    inputs = midas_processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = midas_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # --- FUSION LOGIC AND OUTPUT GENERATION ---
    annotated_frame = frame.copy()
    h, w, _ = frame.shape
    depth_map_resized = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False,
    ).squeeze()

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            depth_value = depth_map_resized[cy, cx].item()
            
            distance_category = ""
            if depth_value < 2500:
                distance_category = "CLOSE"
            elif depth_value < 3500:
                distance_category = "MEDIUM"
            else:
                distance_category = "FAR"
            
            label = f"{class_name} - {distance_category}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ** NEW non-blocking audio alert **
            if distance_category == "CLOSE":
                current_time = time.time()
                print(f"DEBUG: CLOSE object detected. Time since last alert: {current_time - last_alert_time:.2f}s")
                if current_time - last_alert_time > ALERT_COOLDOWN:
                    print(f"DEBUG: Cooldown of {ALERT_COOLDOWN}s passed. Sending alert to TTS queue.")
                    alert_text = f"Warning, {class_name} is close."
                    tts_queue.put(alert_text) # Just add the message to the queue
                    last_alert_time = current_time
    
    # --- Display the final frame ---
    final_h, final_w, _ = annotated_frame.shape
    aspect_ratio = final_w / final_h
    new_width = int(DISPLAY_HEIGHT * aspect_ratio)
    resized_frame = cv2.resize(annotated_frame, (new_width, DISPLAY_HEIGHT))
    cv2.imshow("AuraNav Demo", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- 4. CLEANUP ---
print("Stopping script...")
tts_queue.put(None) # Signal the TTS thread to exit
cap.release()
cv2.destroyAllWindows()
tts_thread.join() # Wait for the TTS thread to finish cleanly
print("Script finished.")