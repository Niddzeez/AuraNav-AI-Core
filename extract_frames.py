import cv2
import os

# --- Configuration ---
INPUT_VIDEO_FOLDER = "F:\Videos\Hostel_Indoors"
OUTPUT_IMAGE_FOLDER = "F:\dataset\images"
FRAME_SKIP = 15  # Saves one frame every 15 frames (e.g., 2 frames per second for a 30fps video)
# ---------------------

# Ensure the output directory exists
if not os.path.exists(OUTPUT_IMAGE_FOLDER):
    os.makedirs(OUTPUT_IMAGE_FOLDER)

# Get a list of all video files in the input folder
video_files = [f for f in os.listdir(INPUT_VIDEO_FOLDER) if f.endswith(('.mp4', '.mov', '.avi'))]

total_frames_saved = 0

# Loop through each video file
for video_file in video_files:
    video_path = os.path.join(INPUT_VIDEO_FOLDER, video_file)
    print(f"--- Processing video: {video_file} ---")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count_this_video = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # End of video

        # Check if it's time to save a frame
        if frame_count % FRAME_SKIP == 0:
            # Create a unique filename for the image
            image_filename = f"{os.path.splitext(video_file)[0]}_frame_{frame_count}.jpg"
            output_path = os.path.join(OUTPUT_IMAGE_FOLDER, image_filename)

            # Save the frame as a JPG image
            cv2.imwrite(output_path, frame)
            saved_count_this_video += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count_this_video} frames from this video.")
    total_frames_saved += saved_count_this_video

print(f"\n--- DONE ---")
print(f"Total frames saved from all videos: {total_frames_saved}")