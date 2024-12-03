# data_utils.py
import numpy as np

def get_subset(pairs, labels, subset_fraction=0.2):
    """
    Extract a subset of the dataset based on a fraction of the total dataset.

    Args:
        pairs (np.ndarray): Array of image pairs.
        labels (np.ndarray): Array of labels corresponding to the pairs.
        subset_fraction (float): Fraction of the dataset to include (e.g., 0.2 for 20%).

    Returns:
        subset_pairs (np.ndarray): Subset of image pairs.
        subset_labels (np.ndarray): Subset of labels.
    """
    if not (0.0 < subset_fraction <= 1.0):
        raise ValueError("subset_fraction must be between 0 and 1.")

    subset_size = int(len(labels) * subset_fraction)
    indices = np.random.choice(len(labels), subset_size, replace=False)
    subset_pairs = pairs[indices]
    subset_labels = labels[indices]

    return subset_pairs, subset_labels
