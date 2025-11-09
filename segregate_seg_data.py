import os
import random
import shutil

# --- CONFIGURATION ---
# The single source folder where ALL your .jpg images and .txt labels are mixed together.
# This is the path you just provided.
SOURCE_FOLDER = r"F:\dataset\gate_to_agh_first\outer\labels\train\frames_deduped"

# The brand new, clean destination folder that will be created.
DEST_BASE_FOLDER = r"F:\Final_Segmentation_Dataset"
# ---------------------


# --- Do not change below this line ---
DEST_IMG_TRAIN = os.path.join(DEST_BASE_FOLDER, "images/train")
DEST_IMG_VAL = os.path.join(DEST_BASE_FOLDER, "images/val")
DEST_LABEL_TRAIN = os.path.join(DEST_BASE_FOLDER, "labels/train")
DEST_LABEL_VAL = os.path.join(DEST_BASE_FOLDER, "labels/val")
SPLIT_RATIO = 0.8

def final_organize():
    print("--- Starting Final Dataset Organization ---")

    if not os.path.exists(SOURCE_FOLDER):
        print(f"FATAL ERROR: Source folder not found at '{SOURCE_FOLDER}'. Please check the path.")
        return

    # Create destination directories
    for path in [DEST_IMG_TRAIN, DEST_IMG_VAL, DEST_LABEL_TRAIN, DEST_LABEL_VAL]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    # Get all image filenames (.jpg) from the single source folder
    all_images = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(('.jpg', '.jpeg'))]
    if not all_images:
        print(f"Error: No image files (.jpg, .jpeg) found in '{SOURCE_FOLDER}'.")
        return
        
    random.shuffle(all_images)

    split_index = int(len(all_images) * SPLIT_RATIO)
    train_filenames = all_images[:split_index]
    val_filenames = all_images[split_index:]

    print(f"\nTotal images found: {len(all_images)}")
    print(f"Assigning {len(train_filenames)} to training set.")
    print(f"Assigning {len(val_filenames)} to validation set.")

    def move_files(filenames, img_dest, label_dest):
        for img_file in filenames:
            base_filename = os.path.splitext(img_file)[0]
            lbl_file = base_filename + ".txt"

            # Source paths are from the single SOURCE_FOLDER
            src_img_path = os.path.join(SOURCE_FOLDER, img_file)
            src_lbl_path = os.path.join(SOURCE_FOLDER, lbl_file)

            # Destination paths are separate
            dest_img_path = os.path.join(img_dest, img_file)
            dest_label_path = os.path.join(label_dest, lbl_file)

            if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
                shutil.move(src_img_path, dest_img_path)
                shutil.move(src_lbl_path, dest_label_path)
            else:
                print(f"Warning: Could not find matching pair for {img_file}. Skipping.")
    
    print("\nMoving training files...")
    move_files(train_filenames, DEST_IMG_TRAIN, DEST_LABEL_TRAIN)
    print("Moving validation files...")
    move_files(val_filenames, DEST_IMG_VAL, DEST_LABEL_VAL)

    print("\n--- DONE ---")

if __name__ == "__main__":
    final_organize()