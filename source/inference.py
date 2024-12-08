import torch
from Siamese_model import SiameseNetwork
from dataloader import load_images_and_masks
import torch.nn.functional as F
import numpy as np


def preprocess_image(img):
    """Convert image to tensor and normalize."""
    img_tensor = torch.tensor(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Batch dimension
    return img_tensor


def pairwise_distance(embedding1, embedding2):
    """Compute pairwise Euclidean distance."""
    # Ensure input tensors have the same dimensions
    assert embedding1.shape == embedding2.shape, "Embeddings must have the same shape."
    return torch.sqrt(torch.sum((embedding1 - embedding2) ** 2, dim=1))


def predict_similarity(model, img1, img2, device="cpu"):
    """
    Predict similarity score between two images using the Siamese model.

    Args:
        model (torch.nn.Module): The Siamese model.
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        device (str): Device to perform inference on ("cpu" or "cuda").

    Returns:
        float: Similarity score between 0 and 1.
    """
    # Preprocess images
    img1_tensor = preprocess_image(img1).to(device)
    img2_tensor = preprocess_image(img2).to(device)

    # Forward pass to compute embeddings
    model.eval()
    with torch.no_grad():
        embedding1, embedding2 = model(img1_tensor, img2_tensor)

    # Normalize embeddings
    embedding1 = F.normalize(embedding1, p=2, dim=1)  # Normalize along feature dimensions
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    print(f"Embedding 1: {embedding1}")
    print(f"Embedding 2: {embedding2}")

    # Compute Euclidean distance
    distance = pairwise_distance(embedding1, embedding2).item()
    print(f"Distance: {distance}")

    # Convert distance to similarity score
    similarity_score = 1 / (1 + distance)
    print(f"Similarity Score: {similarity_score}")

    return similarity_score


def extract_embedding(model, img, device="cpu"):
    """
    Extract feature embeddings from the shared network.

    Args:
        model (torch.nn.Module): The Siamese model.
        img (numpy.ndarray): Input image.
        device (str): Device to perform inference on ("cpu" or "cuda").

    Returns:
        torch.Tensor: Embedding vector.
    """
    img_tensor = preprocess_image(img).to(device)
    with torch.no_grad():
        embedding = model.forward_once(img_tensor)
    return embedding


if __name__ == "__main__":
    model_path = "siamese_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load images and labels
    images, _, labels = load_images_and_masks("./", target_size=(128, 128))
    print(f"Loaded {len(images)} images.")

    if len(images) == 0:
        print("No images were loaded, check the directory")
    else:
        # Load the model with weights_only=True
        model = SiameseNetwork()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Find a group with at least two images
        group_indices = np.where(labels == labels[0])[0]  # All indices for the same group
        if len(group_indices) < 2:
            print("Not enough images in the group to select two distinct ones.")
            img1, img2 = images[group_indices[0]], images[group_indices[0]]  # Same image (fallback)
        else:
            img1, img2 = images[group_indices[0]], images[group_indices[1]]  # Two distinct images

        # Select a different image for comparison
        different_group_indices = np.where(labels != labels[0])[0]
        if len(different_group_indices) == 0:
            print("No images from a different group available for comparison. Using the same group for testing.")
            img3 = img1  # Fallback
        else:
            img3 = images[different_group_indices[0]]

        # Testing similarity scores
        same_group_score = predict_similarity(model, img1, img2, device)
        different_group_score = predict_similarity(model, img1, img3, device)

        print(f"Similarity Score (Same Group): {same_group_score:.4f}")
        print(f"Similarity Score (Different Group): {different_group_score:.4f}")

        embedding1_same = extract_embedding(model, img1, device)
        embedding2_same = extract_embedding(model, img2, device)
        embedding2_diff = extract_embedding(model, img3, device)

        print(f"Embedding Difference (Same Group): {torch.sqrt(torch.sum((embedding1_same - embedding2_same) ** 2)).item()}")
        print(f"Embedding Difference (Different Group): {torch.sqrt(torch.sum((embedding1_same - embedding2_diff) ** 2)).item()}")
