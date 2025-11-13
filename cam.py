import cv2
import requests
import numpy as np
import time

# Replace with your phone IP
URL = "http://192.0.0.2:8080/shot.jpg"

while True:
    try:
        img_resp = requests.get(URL, timeout=1)
        img_arr = np.frombuffer(img_resp.content, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        cv2.imshow("Live Feed (Fast Mode)", frame)

        # Use your AuraNav pipeline here, e.g.:
        # run_models(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)  # 10 FPS approx.

    except Exception as e:
        print("⚠️ Frame error:", e)
        time.sleep(0.5)

cv2.destroyAllWindows()
