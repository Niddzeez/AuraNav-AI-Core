import cv2
import torch
import numpy as np
import requests
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import win32com.client as wincl
import time
import threading
from queue import Queue
import pythoncom

# --- 1. CONFIGURATION ---
# !!! CRITICAL: UPDATE THESE PATHS !!!
YOLO_DETECTION_MODEL_PATH = r'C:\Users\HP\OneDrive\Desktop\AuraNav_sprint\runs\detect\train\weights\best.pt'
YOLO_SEGMENTATION_MODEL_PATH = r'C:\Users\HP\OneDrive\Desktop\AuraNav_sprint\runs\segment\train8\weights\best.pt'

# IP Webcam stream URL (from your phone)
# ⚠️ Ensure this opens in your browser before running the code
IP_WEBCAM_URL = "http://172.20.10.10:8080/shot.jpg"  # Replace with your actual working IP

DISPLAY_HEIGHT = 720
ALERT_COOLDOWN = 8  # seconds


# --- 2. TEXT-TO-SPEECH THREAD ---
def tts_worker(q):
    pythoncom.CoInitialize()
    speak = wincl.Dispatch("SAPI.SpVoice")
    while True:
        text = q.get()
        if text is None:
            break
        speak.Speak(text)
        q.task_done()


# --- 3. INITIALIZATION ---
print("Loading models... This may take a moment.")
tts_queue = Queue()
last_alert_time = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Start TTS thread
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
    print("✅ All AI models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()


# --- 4. LIVE FRAME FETCH FUNCTION ---
def get_live_frame():
    """Fetch a single frame from the IP Webcam stream."""
    try:
        response = requests.get(IP_WEBCAM_URL, timeout=2)
        img_arr = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print("⚠️ Frame fetch error:", e)
        return None


print("\n🎥 Starting live camera processing... Press 'q' to quit.")


# --- 5. MAIN PROCESSING LOOP ---
while True:
    frame = get_live_frame()
    if frame is None:
        continue

    # --- STEP A: Run All Three Models ---
    yolo_det_results = yolo_detection_model(frame, verbose=False)
    seg_results = yolo_segmentation_model(frame, verbose=False)

    inputs = midas_processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = midas_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # --- STEP B: Fusion & Visualization ---
    annotated_frame = frame.copy()
    h, w, _ = frame.shape

    # Path segmentation
    walkable_mask = np.zeros((h, w), dtype=np.uint8)
    obstacle_mask = np.zeros((h, w), dtype=np.uint8)

    if seg_results[0].masks is not None:
        for i, cls in enumerate(seg_results[0].boxes.cls):
            mask_raw = seg_results[0].masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_raw, (w, h)).astype(np.uint8)

            if int(cls) == 1:  # 'walkable'
                walkable_mask = cv2.bitwise_or(walkable_mask, mask_resized)
            else:
                obstacle_mask = cv2.bitwise_or(obstacle_mask, mask_resized)

        # Green overlay for walkable area
        green_overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
        green_overlay[:, :] = (0, 255, 0)
        walkable_area_only = cv2.bitwise_and(green_overlay, green_overlay, mask=walkable_mask)
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, walkable_area_only, 0.4, 0)

    # Depth processing
    depth_map_resized = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Object detection results
    for result in yolo_det_results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            class_id = int(box.cls[0])
            class_name = yolo_detection_model.names[class_id]

            cx = int((x1 + x2) / 2)
            is_on_path = walkable_mask[y2 - 5, cx] > 0 if y2 - 5 < h else False

            depth_value = depth_map_resized[int((y1 + y2) / 2), cx].item()
            if depth_value < 2500:
                distance_category = "CLOSE"
            elif depth_value < 3500:
                distance_category = "MEDIUM"
            else:
                distance_category = "FAR"

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

    # --- STEP C: Generic Obstacle Warning ---
    roi_h_start = int(h * 0.6)
    roi_w_start = int(w * 0.33)
    roi_w_end = int(w * 0.66)

    danger_zone_mask = obstacle_mask[roi_h_start:h, roi_w_start:roi_w_end]
    danger_zone_depth = depth_map_resized[roi_h_start:h, roi_w_start:roi_w_end]

    obstacle_percentage = (np.sum(danger_zone_mask) / danger_zone_mask.size) * 100

    if obstacle_percentage > 50:
        avg_depth = danger_zone_depth.mean().item()
        if avg_depth < 2500:
            current_time = time.time()
            if current_time - last_alert_time > ALERT_COOLDOWN:
                tts_queue.put("Warning, obstacle ahead.")
                last_alert_time = current_time

    # --- STEP D: Display Output ---
    aspect_ratio = w / h
    new_width = int(DISPLAY_HEIGHT * aspect_ratio)
    resized_frame = cv2.resize(annotated_frame, (new_width, DISPLAY_HEIGHT))

    cv2.imshow("AuraNav Live - Real-Time Obstacle Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)  # prevent overload

# --- 6. CLEANUP ---
print("Stopping script...")
tts_queue.put(None)
cv2.destroyAllWindows()
tts_thread.join()
print("✅ Script finished.")
