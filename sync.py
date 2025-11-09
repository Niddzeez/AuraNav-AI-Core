import os

# --- CONFIGURATION ---
# Path to the folder with your final .txt label files
LABEL_FOLDER = r"F:\dataset\inner-hostel-night\obj_train_data"

# Path to the folder with ALL of your .jpg image files
IMAGE_FOLDER = r"F:\dataset\raw_images"

# --- SAFETY FEATURE ---
# Set to True to only PRINT which files would be deleted, without actually deleting them.
# Set to False to perform the actual deletion.
DRY_RUN = False
# --------------------

def sync_folders():
    print("--- Starting Folder Sync ---")
    
    # Get the base names of all label files (e.g., "frame_45" from "frame_45.txt")
    # Using a set for very fast lookups
    label_basenames = {os.path.splitext(f)[0] for f in os.listdir(LABEL_FOLDER) if f.endswith('.txt')}
    
    print(f"Found {len(label_basenames)} unique labels in '{LABEL_FOLDER}'")

    # Get all image files from the image folder
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images in '{IMAGE_FOLDER}'")
    
    deleted_count = 0
    
    # Loop through all image files
    for img_file in image_files:
        # Get the base name of the image file
        img_basename = os.path.splitext(img_file)[0]
        
        # Check if the image's base name does NOT exist in our set of label names
        if img_basename not in label_basenames:
            file_to_delete = os.path.join(IMAGE_FOLDER, img_file)
            
            if DRY_RUN:
                print(f"[Dry Run] Would delete orphan image: {file_to_delete}")
            else:
                try:
                    os.remove(file_to_delete)
                    print(f"Deleted orphan image: {file_to_delete}")
                except Exception as e:
                    print(f"Error deleting {file_to_delete}: {e}")

            deleted_count += 1
            
    print("\n--- Sync Complete ---")
    if DRY_RUN:
        print(f"Found {deleted_count} orphan images that would be deleted. To delete them, set DRY_RUN to False and run again.")
    else:
        print(f"Successfully deleted {deleted_count} orphan images.")

if __name__ == "__main__":
    sync_folders()