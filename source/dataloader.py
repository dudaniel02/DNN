import os
import cv2
import numpy as np
import os

def normalize_filename(filename):
    """Normalize filenames by stripping whitespaces and converting to lowercase."""
    return filename.strip().lower()

def load_images_and_masks(group_folder, target_size=(128, 128)):
    """Load images and masks from a folder."""
    images, masks, labels = [], [], []
    image_folder = os.path.join(group_folder, "fish_image")
    mask_folder = os.path.join(group_folder, "mask_image")

    if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
        print(f"Folders {image_folder} or {mask_folder} do not exist.")
        return np.array(images), np.array(masks), np.array(labels)

    fish_groups = sorted(os.listdir(image_folder))
    mask_groups = sorted(os.listdir(mask_folder))

    for fish_group, mask_group in zip(fish_groups, mask_groups):
        fish_group_path = os.path.join(image_folder, fish_group)
        mask_group_path = os.path.join(mask_folder, mask_group)

        if os.path.isdir(fish_group_path) and os.path.isdir(mask_group_path):
            fish_files = os.listdir(fish_group_path)
            mask_files = os.listdir(mask_group_path)

            for fish_file in fish_files:
                if fish_file.startswith("fish_") and fish_file.endswith(".png"):
                    # Normalize fish and mask filenames
                    fish_filename = normalize_filename(fish_file)
                    mask_filename = fish_filename.replace("fish_", "mask_")  # Replace "fish_" with "mask_"

                    # Full paths to fish image and mask
                    fish_file_path = os.path.join(fish_group_path, fish_filename)
                    mask_file_path = os.path.join(mask_group_path, mask_filename)

                    if not os.path.exists(mask_file_path):
                        print(f"Mask for {fish_file} not found in {mask_group_path}")
                        continue

                    # Load fish image and mask
                    img = cv2.imread(fish_file_path)
                    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)

                    if img is None or mask is None:
                        print(f"Failed to load {fish_file_path} or {mask_file_path}")
                        continue

                    # Resize and process
                    img = cv2.resize(img, target_size)
                    mask = cv2.resize(mask, target_size)

                    # Use the numeric part of the filename as the label (e.g., "000000009598_05281")
                    fish_id = fish_filename[5:-4]  # Strip "fish_" prefix and ".png" suffix
                    labels.append(fish_id)

                    images.append(img)
                    masks.append(mask)

    return np.array(images), np.array(masks), np.array(labels)

if __name__ == "__main__":
    images, masks, labels = load_images_and_masks("data/raw")
    print(f"Loaded {len(images)} images and {len(masks)} masks.")
