import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=256):
        super(SiameseNetwork, self).__init__()
        self.embedding_size = embedding_size

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

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(512, self.embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.embedding_size),
            nn.Dropout(0.5)
        )

        self._initialize_weights()

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
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.cnn(x)
        embedding = self.fc(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
