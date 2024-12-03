import numpy as np
import random
from collections import defaultdict

def create_pairs(images, labels, max_pairs=10000):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    positive_pairs = []
    negative_pairs = []

    labels_set = list(set(labels))
    num_labels = len(labels_set)

    # Generate positive pairs
    for _ in range(max_pairs // 2):
        label = random.choice(labels_set)
        idx1, idx2 = random.sample(label_to_indices[label], 2)
        positive_pairs.append([images[idx1], images[idx2], 1])

    # Generate negative pairs
    for _ in range(max_pairs // 2):
        label1, label2 = random.sample(labels_set, 2)
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
