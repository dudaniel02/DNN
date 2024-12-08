from tqdm import tqdm
import os
import cv2
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
import torch
from collections import defaultdict

def normalize_filename(filename):
    return filename.strip().lower()

def augment_image(img):
    from torchvision import transforms
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    img_augmented = augmentation(img)
    return (img_augmented.numpy() * 255).astype(np.uint8)

def load_images_and_masks(group_folder, target_size=(128, 128)):
    images, masks, labels = [], [], []
    image_folder = os.path.join(group_folder, "fish_image")
    mask_folder = os.path.join(group_folder, "mask_image")

    if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
        print(f"Folders {image_folder} or {mask_folder} do not exist.")
        return np.array(images), np.array(masks), np.array(labels)

    fish_groups = sorted(os.listdir(image_folder))
    mask_groups = sorted(os.listdir(mask_folder))

    for fish_group, mask_group in tqdm(zip(fish_groups, mask_groups), desc="Loading images", total=len(fish_groups)):
        fish_group_path = os.path.join(image_folder, fish_group)
        mask_group_path = os.path.join(mask_folder, mask_group)

        if os.path.isdir(fish_group_path) and os.path.isdir(mask_group_path):
            fish_files = os.listdir(fish_group_path)
            mask_files = os.listdir(mask_group_path)

            for fish_file in fish_files:
                if fish_file.startswith("fish_") and fish_file.endswith(".png"):
                    fish_filename = normalize_filename(fish_file)
                    mask_filename = fish_filename.replace("fish_", "mask_")

                    fish_file_path = os.path.join(fish_group_path, fish_filename)
                    mask_file_path = os.path.join(mask_group_path, mask_filename)

                    if not os.path.exists(mask_file_path):
                        print(f"Mask for {fish_file} not found in {mask_group_path}")
                        continue

                    img = cv2.imread(fish_file_path)
                    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)

                    if img is None or mask is None:
                        print(f"Failed to load {fish_file_path} or {mask_file_path}")
                        continue

                    if len(img.shape) == 2 or img.shape[2] == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                    img = cv2.resize(img, target_size)
                    mask = cv2.resize(mask, target_size)

                    fish_id = fish_filename[5:-4]
                    labels.append(fish_id)

                    images.append(img)
                    masks.append(mask)

    print(f"Loaded {len(images)} images and {len(masks)} masks.")
    return np.array(images), np.array(masks), np.array(labels)

def create_balanced_pairs(images, labels, max_pairs=10000):
    # Map labels to indices of images
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    positive_pairs = []
    negative_pairs = []

    # Get unique labels
    labels_set = list(set(labels))

    # Generate positive pairs: Different fish, same label
    for _ in range(max_pairs // 2):
        # Select a random label
        label = random.choice(labels_set)
        indices = label_to_indices[label]

        # Ensure there are at least two different images for the label
        if len(indices) < 2:
            continue
        
        # Select two different images from the same group
        idx1, idx2 = random.sample(indices, 2)
        positive_pairs.append([images[idx1], images[idx2], 1])

    # Generate negative pairs: Fish from different labels
    for _ in range(max_pairs // 2):
        label1, label2 = random.sample(labels_set, 2)

        # Ensure there are images for both labels
        if len(label_to_indices[label1]) == 0 or len(label_to_indices[label2]) == 0:
            continue
        
        # Select one random image from each group
        idx1 = random.choice(label_to_indices[label1])
        idx2 = random.choice(label_to_indices[label2])
        negative_pairs.append([images[idx1], images[idx2], 0])

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    # Split into image pairs and labels
    image_pairs = np.array([(pair[0], pair[1]) for pair in all_pairs])
    pair_labels = np.array([pair[2] for pair in all_pairs])

    return image_pairs, pair_labels

class SiamesePairDataset(Dataset):
    def __init__(self, image_pairs, pair_labels):
        self.image_pairs = image_pairs
        self.pair_labels = pair_labels

    def __len__(self):
        return len(self.pair_labels)

    def __getitem__(self, idx):
        img1, img2 = self.image_pairs[idx]
        label = self.pair_labels[idx]

        img1 = torch.tensor(img1).float().permute(2, 0, 1) / 255.0
        img2 = torch.tensor(img2).float().permute(2, 0, 1) / 255.0
        label = torch.tensor(label).float()

        return img1, img2, label
