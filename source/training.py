import torch
import torch.nn as nn
import torch.optim as optim
from Siamese_model import SiameseNetwork
from pair_generator import create_pairs
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from tqdm import tqdm
from data_utils import get_subset



# Ensure reproducibility
torch.manual_seed(42)

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

# Custom Dataset
class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        label = self.labels[idx]
        return torch.tensor(img1).float(), torch.tensor(img2).float(), torch.tensor(label).float()



def train_model(model, train_loader, epochs=10, learning_rate=0.001, margin=1.0, device="cpu"):
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0

        # Add a progress bar for the epoch
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for img1, img2, label in train_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                optimizer.zero_grad()
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                euclidean_distance = F.pairwise_distance(output1, output2)
                predictions = (euclidean_distance < margin).float()
                correct += (predictions == label).sum().item()
                total_samples += label.size(0)

                # Update the progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        accuracy = correct / total_samples
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the model
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "outputs/checkpoints/siamese_model.pth")
    print("Model saved to outputs/checkpoints/siamese_model.pth")

def evaluate_model(model, test_loader, margin=1.0, device="cpu"):
    criterion = ContrastiveLoss(margin=margin)
    model.to(device)
    model.eval()

    total_loss = 0
    correct = 0
    total_samples = 0

    # Add a progress bar for evaluation
    with tqdm(total=len(test_loader), desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for img1, img2, label in test_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                total_loss += loss.item()

                euclidean_distance = F.pairwise_distance(output1, output2)
                predictions = (euclidean_distance < margin).float()
                correct += (predictions == label).sum().item()
                total_samples += label.size(0)

                # Update the progress bar
                pbar.update(1)

    accuracy = correct / total_samples
    print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Main block for testing
if __name__ == "__main__":
    import numpy as np



    # Configuration
    use_subset = True  # Flag to enable/disable subset usage
    subset_fraction = 0.05  # Use 20% of the dataset as a subset
    epochs = 5
    margin = 1.0
    batch_size = 32
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy data for testing
    dummy_images = np.random.rand(1000, 3, 128, 128)  # Shape (N, C, H, W)
    dummy_labels = np.array([i // 10 for i in range(1000)])  # 10 classes, 100 images each
    pairs, labels = create_pairs(dummy_images, dummy_labels)

    # Optionally use a subset of the data
    if use_subset:
        print(f"Using {int(subset_fraction * 100)}% of the dataset as a subset")
        pairs, labels = get_subset(pairs, labels, subset_fraction=subset_fraction)

    # Prepare dataset and dataloader
    train_dataset = SiameseDataset(pairs, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = SiameseNetwork(embedding_size=256)

    # Train the model
    train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate, margin=margin, device=device)

    # Prepare test data (use subset if needed)
    test_pairs, test_labels = create_pairs(dummy_images, dummy_labels)
    if use_subset:
        print(f"Using {int(subset_fraction * 100)}% of the dataset for evaluation")
        test_pairs, test_labels = get_subset(test_pairs, test_labels, subset_fraction=subset_fraction)

    test_dataset = SiameseDataset(test_pairs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    evaluate_model(model, test_loader, margin=margin, device=device)
