import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=256):  # Use 256 to match the saved model
        super(SiameseNetwork, self).__init__()
        self.embedding_size = embedding_size

        # Convolutional Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten()
        )

        # Fully Connected Layer for Embedding
        self.fc = nn.Sequential(
            nn.Linear(512, self.embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.embedding_size),
            nn.Dropout(0.5)  # Regularization
        )

    

        self._initialize_weights()

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        label = self.labels[idx]
        img1 = torch.tensor(img1 / 255.0).float().permute(2, 0, 1)  # Normalize and permute
        img2 = torch.tensor(img2 / 255.0).float().permute(2, 0, 1)
        return img1, img2, torch.tensor(label).float()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_once(self, x):
        features = self.cnn(x)
        embedding = self.fc(features)
        embedding = F.normalize(embedding, p=2, dim=1)  # Normalize embeddings
        return embedding

    def forward(self, input1, input2):
        """Process both inputs and return their embeddings."""
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


if __name__ == "__main__":
    # Instantiate the model
    model = SiameseNetwork(embedding_size=256)  # Example with a smaller embedding size

    # Test with dummy data
    input_a = torch.randn(4, 3, 128, 128)  # Batch size = 1, channels = 3, height = 128, width = 128
    input_b = torch.randn(4, 3, 128, 128)

    output1, output2 = model(input_a, input_b)
    print(f"Output1 Shape: {output1.shape}, Output2 Shape: {output2.shape}")
