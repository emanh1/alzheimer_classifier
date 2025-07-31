import torch.nn as nn

class Tiny3DCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # (30, 124, 248)

            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # (15, 62, 124)

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)  # → (32, 1, 1, 1)
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
