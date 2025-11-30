# training/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """
    Simple convolutional neural network for MNIST (28x28 grayscale).
    """

    def __init__(self):
        super().__init__()
        # Input: (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> (32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (64, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)                            # -> (64, 14, 14)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 digit classes (0–9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x