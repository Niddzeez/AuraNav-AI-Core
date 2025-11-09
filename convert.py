import os
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
# The base folder where your 'labels' folder is located
BASE_DATA_FOLDER = r"F:\dataset\segmentation_dataset"

# Define the colors CVAT used and the class IDs you want
# IMPORTANT: You may need to adjust these RGB values if your CVAT uses different colors.
# (R, G, B) -> class_id
# You can find the exact RGB values by using a color picker in MS Paint or Photoshop on one of your masks.
COLOR_MAP = {
    (128, 64, 128): 0,  # This is a common color for 'road' or 'path' in CamVid format
    (64, 0, 0): 1      # This is a common color for an obstacle/void
    # Add other color-to-class mappings here if you have more
}
# ---------------------

def convert_masks_in_folder(folder_path):
    converted_count = 0
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found at {folder_path}. Skipping.")
        return 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            mask_path = os.path.join(folder_path, filename)
            
            # Open the color-coded mask
            color_mask = Image.open(mask_path).convert("RGB")
            color_mask_np = np.array(color_mask)
            
            # Create a new, empty grayscale mask (all zeros)
            index_mask_np = np.zeros((color_mask_np.shape[0], color_mask_np.shape[1]), dtype=np.uint8)
            
            # Convert each color to its corresponding class index
            for color, class_id in COLOR_MAP.items():
                # Find all pixels that match the current color
                matches = (color_mask_np == color).all(axis=-1)
                # Set those pixels in the new mask to the class ID
                index_mask_np[matches] = class_id
            
            # Save the new index mask, overwriting the old one
            index_mask_img = Image.fromarray(index_mask_np)
            index_mask_img.save(mask_path)
            converted_count += 1
            
    return converted_count

def main():
    print("--- Starting Mask Conversion ---")
    
    # Define the folders to process
    train_labels_path = os.path.join(BASE_DATA_FOLDER, "labels/train")
    val_labels_path = os.path.join(BASE_DATA_FOLDER, "labels/val")
    
    print(f"\nProcessing training masks in: {train_labels_path}")
    train_count = convert_masks_in_folder(train_labels_path)
    print(f"Converted {train_count} training masks.")
    
    print(f"\nProcessing validation masks in: {val_labels_path}")
    val_count = convert_masks_in_folder(val_labels_path)
    print(f"Converted {val_count} validation masks.")
    
    print("\n--- DONE ---")

if __name__ == "__main__":
    main()