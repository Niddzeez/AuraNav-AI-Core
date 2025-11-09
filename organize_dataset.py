import os
import random
import shutil

# --- CONFIGURATION ---
# 1. Path to your NEW folder containing the new images AND labels
SOURCE_FOLDER = r"F:\dataset\gate_to_agh_first\extendend\obj_train_data\frames_deduped"

# 2. Path to your EXISTING dataset's base folder (where 'images' and 'labels' are)
DEST_BASE_FOLDER = r"F:\dataset" 

# 3. The train/validation split for the NEW data
SPLIT_RATIO = 0.8
# ---------------------

# --- Do not change below this line ---
DEST_IMG_TRAIN = os.path.join(DEST_BASE_FOLDER, "images/train")
DEST_IMG_VAL = os.path.join(DEST_BASE_FOLDER, "images/val")
DEST_LBL_TRAIN = os.path.join(DEST_BASE_FOLDER, "labels/train")
DEST_LBL_VAL = os.path.join(DEST_BASE_FOLDER, "labels/val")

def add_new_files():
    print("--- Starting to Add New Data to Dataset ---")

    # Check if paths exist
    if not os.path.exists(SOURCE_FOLDER):
        print(f"FATAL ERROR: Source folder not found at '{SOURCE_FOLDER}'")
        return
    if not os.path.exists(DEST_IMG_TRAIN):
        print(f"FATAL ERROR: Destination folder not found at '{DEST_IMG_TRAIN}'")
        return

    # Get all image filenames (.jpg) from the NEW source folder
    all_new_images = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_new_images:
        print(f"Error: No image files found in '{SOURCE_FOLDER}'.")
        return
        
    random.shuffle(all_new_images)

    split_index = int(len(all_new_images) * SPLIT_RATIO)
    train_filenames = all_new_images[:split_index]
    val_filenames = all_new_images[split_index:]

    print(f"\nFound {len(all_new_images)} new images to add.")
    print(f"Copying {len(train_filenames)} to training set.")
    print(f"Copying {len(val_filenames)} to validation set.")

    # Function to COPY files (safer than moving)
    def copy_files(filenames, img_dest, label_dest):
        copied_count = 0
        for img_file in filenames:
            base_filename = os.path.splitext(img_file)[0]
            lbl_file = base_filename + ".txt"

            # Source paths are from the NEW folder
            src_img_path = os.path.join(SOURCE_FOLDER, img_file)
            src_lbl_path = os.path.join(SOURCE_FOLDER, lbl_file)

            # Destination paths are the EXISTING folders
            dest_img_path = os.path.join(img_dest, img_file)
            dest_label_path = os.path.join(label_dest, lbl_file)

            if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
                shutil.copy(src_img_path, dest_img_path)
                shutil.copy(src_lbl_path, dest_label_path)
                copied_count += 1
            else:
                print(f"Warning: Could not find matching pair for {img_file}. Skipping.")
        return copied_count

    print("\nCopying new training files...")
    train_copied = copy_files(train_filenames, DEST_IMG_TRAIN, DEST_LBL_TRAIN)
    print(f"Copied {train_copied} new training image/label pairs.")

    print("\nCopying new validation files...")
    val_copied = copy_files(val_filenames, DEST_IMG_VAL, DEST_LBL_VAL)
    print(f"Copied {val_copied} new validation image/label pairs.")

    print("\n--- DONE ---")
    print(f"Your training and validation sets have been successfully augmented with {train_copied + val_copied} new files.")

if __name__ == "__main__":
    add_new_files()