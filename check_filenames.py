import os

IMG_DIR = r"F:\dataset\segmentation_dataset\images\train"
LBL_DIR = r"F:\dataset\segmentation_dataset\labels\train"

print("--- Checking for filename mismatches ---")

img_basenames = {os.path.splitext(f)[0] for f in os.listdir(IMG_DIR)}
lbl_basenames = {os.path.splitext(f)[0] for f in os.listdir(LBL_DIR)}

missing_in_labels = img_basenames - lbl_basenames
missing_in_images = lbl_basenames - img_basenames

if not missing_in_labels and not missing_in_images:
    print("Success! All filenames are perfectly matched.")
else:
    if missing_in_labels:
        print(f"\nERROR: Found {len(missing_in_labels)} images that are MISSING a corresponding label file:")
        for name in list(missing_in_labels)[:5]:
            print(f"  - {name}.jpg")

    if missing_in_images:
        print(f"\nERROR: Found {len(missing_in_images)} label files that are MISSING a corresponding image file:")
        for name in list(missing_in_images)[:5]:
            print(f"  - {name}.png")