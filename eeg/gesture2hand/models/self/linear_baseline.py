import torch
from torch import nn


class EEGLinearBaseline(nn.Module):
    """
    Bandpower features (B, T, 84) -> mean pool -> logit (B, 1)
    """

    def __init__(
        self, num_features: int = 84, num_classes: int = 4, dropout: float = 0.3
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1)  # (B, 84)
        return self.head(pooled)
