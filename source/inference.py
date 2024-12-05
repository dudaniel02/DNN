import torch
from Siamese_model import SiameseNetwork
from dataloader import load_images_and_masks
from torch.nn.functional import pairwise_distance


import torch
import torch.nn.functional as F

def preprocess_image(img):
    """Convert image to tensor and normalize."""
    img_tensor = torch.tensor(img).float().permute(2, 0, 1) / 255.0  # Example preprocessing
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def pairwise_distance(embedding1, embedding2):
    """Compute pairwise Euclidean distance."""
    return F.pairwise_distance(embedding1, embedding2)

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

    # Compute Euclidean distance
    distance = pairwise_distance(embedding1, embedding2).item()

    # Convert distance to similarity score
    similarity_score = 1 / (1 + distance)  # Inverse relationship: lower distance = higher similarity score

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

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load images and labels
    images, _, labels = load_images_and_masks("./", target_size=(128, 128))
    print(f"Loaded {len(images)} images.")

    if len(images) == 0:
        print("No images were loaded. Please check the directory and file paths.")
    else:
        # Initialize and load the model
        model = SiameseNetwork()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        img1, img2 = images[0], images[0]  # Same image
        img3 = images[1]  # Different image

        print("Testing Similarity:")
        same_score = predict_similarity(model, img1, img2, device)  
        diff_score = predict_similarity(model, img1, img3, device)  

        print(f"Similarity Score (Same Image): {same_score:.4f}")
        print(f"Similarity Score (Different Image): {diff_score:.4f}")

        embedding1_same = extract_embedding(model, img1, device)
        embedding2_same = extract_embedding(model, img2, device)
        embedding2_diff = extract_embedding(model, img3, device)

        print(f"Embedding Difference (Same Pair): {torch.abs(embedding1_same - embedding2_same).sum().item()}")
        print(f"Embedding Difference (Different Pair): {torch.abs(embedding1_same - embedding2_diff).sum().item()}")
