import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from Siamese_model import SiameseNetwork, ContrastiveLoss
from dataloader import load_images_and_masks, create_balanced_pairs, SiamesePairDataset 
import numpy as np

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001, margin=1.0, device="cpu"):
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        correct = 0
        total_samples = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for img1, img2, label in train_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                optimizer.zero_grad()
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()

                total_loss += loss.item()
                euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
                predictions = (euclidean_distance < margin).float()
                correct += (predictions == label).sum().item()
                total_samples += label.size(0)

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        scheduler.step()
        accuracy = correct / total_samples
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()
        
        print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    os.makedirs("./", exist_ok=True)
    torch.save(model.state_dict(), "siamese_model.pth")
    print("Model saved siamese_model.pth")

def evaluate_model(model, test_loader, margin=0.5, device="cpu"):
    criterion = ContrastiveLoss(margin=margin)
    model.to(device)
    model.eval()

    total_loss = 0
    correct = 0
    total_samples = 0

    with tqdm(total=len(test_loader), desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for img1, img2, label in test_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                total_loss += loss.item()

                euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
                predictions = (euclidean_distance < margin).float()
                correct += (predictions == label).sum().item()
                total_samples += label.size(0)

                pbar.update(1)

    accuracy = correct / total_samples
    print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Load and preprocess data
    group_folder = "./"
    images, masks, labels = load_images_and_masks(group_folder)

    # Convert labels to integers
    unique_labels = {label: idx for idx, label in enumerate(set(labels))}
    labels_int = np.array([unique_labels[label] for label in labels])

    # Create balanced pairs
    image_pairs, pair_labels = create_balanced_pairs(images, labels_int, max_pairs=10000)

    # Split data into training and validation
    split_ratio = 0.8
    split_idx = int(len(image_pairs) * split_ratio)
    train_image_pairs, val_image_pairs = image_pairs[:split_idx], image_pairs[split_idx:]
    train_pair_labels, val_pair_labels = pair_labels[:split_idx], pair_labels[split_idx:]

    # Create datasets and dataloaders
    train_dataset = SiamesePairDataset(train_image_pairs, train_pair_labels)
    val_dataset = SiamesePairDataset(val_image_pairs, val_pair_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize and train the model
    model = SiameseNetwork(embedding_size=256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.00001, margin=1.0, device=device)

    # Evaluate the model (using the validation set as the test set for simplicity)
    evaluate_model(model, val_loader, margin=1.0, device=device)
