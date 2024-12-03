import torch
from Siamese_model import SiameseNetwork
from dataloader import load_images_and_masks
from torch.nn.functional import pairwise_distance


def preprocess_image(img):
    """Preprocess image for model input (normalize and convert to tensor)."""
    img_tensor = torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float()  # Normalize and reshape
    return img_tensor


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
    with torch.no_grad():
        embedding1, embedding2 = model(img1_tensor, img2_tensor)

    # Compute Euclidean distance
    distance = pairwise_distance(embedding1, embedding2)

    # Convert distance to similarity score
    similarity_score = torch.sigmoid(-distance).item()  # Inverse distance normalized to 0-1

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
    # Path to the saved model
    model_path = "outputs/checkpoints/siamese_model.pth"

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load images and labels
    images, _, labels = load_images_and_masks("data/raw", target_size=(128, 128))

    # Initialize and load the model
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Test on same and different pairs
    img1, img2 = images[0], images[0]  # Same image
    img3 = images[1]  # Different image

    print("Testing Similarity:")
    same_score = predict_similarity(model, img1, img2, device)  # Same image
    diff_score = predict_similarity(model, img1, img3, device)  # Different images

    print(f"Similarity Score (Same Image): {same_score:.4f}")
    print(f"Similarity Score (Different Image): {diff_score:.4f}")

    print("\nTesting Feature Embeddings:")
    embedding1_same = extract_embedding(model, img1, device)
    embedding2_same = extract_embedding(model, img2, device)
    embedding2_diff = extract_embedding(model, img3, device)

    print(f"Embedding (Same Image): {embedding1_same}")
    print(f"Embedding (Different Image): {embedding2_diff}")
    print(f"Embedding Difference (Same Pair): {torch.abs(embedding1_same - embedding2_same).sum().item()}")
    print(f"Embedding Difference (Different Pair): {torch.abs(embedding1_same - embedding2_diff).sum().item()}")
