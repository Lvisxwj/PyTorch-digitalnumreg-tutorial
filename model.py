"""Shared CNN model for 28x28 grayscale digit recognition."""
from torch import nn


class DigitCNN(nn.Module):
    architecture = "digit-cnn-v1"

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x.reshape(-1, 1, 28, 28)))


NeuralNetwork = DigitCNN
