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
# !!! CRITICAL: UPDATE THESE THREE PATHS !!!
YOLO_DETECTION_MODEL_PATH = r'C:\Users\HP\OneDrive\Desktop\AuraNav_sprint\runs\detect\train\weights\best.pt'
YOLO_SEGMENTATION_MODEL_PATH = r'C:\Users\HP\OneDrive\Desktop\AuraNav_sprint\runs\segment\train8\weights\best.pt'
VIDEO_FILE_PATH = r"F:\Videos\Hostel_Indoors\VID_20251011_224928.mp4"

# Display and alert settings
DISPLAY_HEIGHT = 720
ALERT_COOLDOWN = 8 # seconds
# ---------------------

# --- TTS Worker Function (runs in the background) ---
def tts_worker(q):
    pythoncom.CoInitialize()
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
tts_thread.daemon = True
tts_thread.start()
print("TTS worker thread started.")

# Load AI models
try:
    yolo_detection_model = YOLO(YOLO_DETECTION_MODEL_PATH)
    yolo_segmentation_model = YOLO(YOLO_SEGMENTATION_MODEL_PATH)
    midas_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    midas_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
    print("All AI models loaded successfully.")
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

    # --- STEP A: Run All Three Models in Parallel ---
    yolo_det_results = yolo_detection_model(frame, verbose=False)
    seg_results = yolo_segmentation_model(frame, verbose=False)
    
    inputs = midas_processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = midas_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # --- STEP B: FUSION LOGIC AND OUTPUT GENERATION ---
    annotated_frame = frame.copy()
    h, w, _ = frame.shape
    
    # --- Part 1: Process Path Segmentation ---
    walkable_mask = np.zeros((h, w), dtype=np.uint8)
    obstacle_mask = np.zeros((h, w), dtype=np.uint8) # NEW: We'll also build an obstacle mask
    if seg_results[0].masks is not None:
        for i, cls in enumerate(seg_results[0].boxes.cls):
            mask_raw = seg_results[0].masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_raw, (w, h)).astype(np.uint8)
            
            if int(cls) == 1: # Class 1 is 'walkable'
                walkable_mask = cv2.bitwise_or(walkable_mask, mask_resized)
            else: # Everything else is an obstacle
                obstacle_mask = cv2.bitwise_or(obstacle_mask, mask_resized)
        
        # Visualize the walkable path with a green overlay
        green_overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
        green_overlay[:, :] = (0, 255, 0)
        walkable_area_only = cv2.bitwise_and(green_overlay, green_overlay, mask=walkable_mask)
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, walkable_area_only, 0.4, 0)

    # --- Part 2: Process Object Detections ---
    depth_map_resized = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False,
    ).squeeze()

    for result in yolo_det_results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            class_id = int(box.cls[0])
            class_name = yolo_detection_model.names[class_id]
            
            cx = int((x1 + x2) / 2)
            
            # --- UPGRADE #1: Check the BOTTOM of the box, not the center ---
            # This is more reliable for determining if an object is on the path.
            is_on_path = walkable_mask[y2 - 5, cx] > 0 # Check 5 pixels up from the bottom
            
            depth_value = depth_map_resized[int((y1 + y2) / 2), cx].item()
            
            distance_category = ""
            if depth_value < 2500: distance_category = "CLOSE"
            elif depth_value < 3500: distance_category = "MEDIUM"
            else: distance_category = "FAR"
            
            label = f"{class_name} - {distance_category}"
            color = (0, 255, 0) if is_on_path else (0, 0, 255)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if distance_category == "CLOSE" and is_on_path:
                current_time = time.time()
                if current_time - last_alert_time > ALERT_COOLDOWN:
                    alert_text = f"Warning, {class_name} is close on your path."
                    tts_queue.put(alert_text)
                    last_alert_time = current_time

    # --- UPGRADE #2: Generic Obstacle Detection ---
    # Define a "danger zone" in the middle of the bottom of the screen
    roi_h_start = int(h * 0.6)
    roi_w_start = int(w * 0.33)
    roi_w_end = int(w * 0.66)
    
    danger_zone_mask = obstacle_mask[roi_h_start:h, roi_w_start:roi_w_end]
    danger_zone_depth = depth_map_resized[roi_h_start:h, roi_w_start:roi_w_end]

    # Calculate the percentage of the danger zone that is an obstacle
    obstacle_percentage = (np.sum(danger_zone_mask) / danger_zone_mask.size) * 100
    
    if obstacle_percentage > 50: # If more than 50% of the zone is an obstacle
        avg_depth = danger_zone_depth.mean().item()
        if avg_depth < 2500: # And the average depth is "CLOSE"
            current_time = time.time()
            if current_time - last_alert_time > ALERT_COOLDOWN:
                tts_queue.put("Warning, obstacle ahead.")
                last_alert_time = current_time
    
    # --- STEP D: Display the final frame ---
    aspect_ratio = w / h
    new_width = int(DISPLAY_HEIGHT * aspect_ratio)
    resized_frame = cv2.resize(annotated_frame, (new_width, DISPLAY_HEIGHT))
    
    cv2.imshow("AuraNav Demo - Final Prototype", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- 4. CLEANUP ---
print("Stopping script...")
tts_queue.put(None)
cap.release()
cv2.destroyAllWindows()
tts_thread.join()
print("Script finished.")