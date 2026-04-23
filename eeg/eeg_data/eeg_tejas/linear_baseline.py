"""
Two models for EEG -> hand open/closed:

1. EEGLinearBaseline — mean-pooled bandpower (existing 71% baseline)
2. EEGTemporalModel — processes bandpower dynamics over time with
   a small transformer, so it can learn temporal patterns like
   mu/beta desynchronization onset and rebound.
"""

import math
import torch
import torch.nn as nn


class EEGLinearBaseline(nn.Module):
    """
    Bandpower features (B, T, 84) -> mean pool -> logit (B, 1)
    """

    def __init__(self, num_features: int = 84, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1)  # (B, 84)
        return self.head(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EEGTemporalModel(nn.Module):
    """
    Bandpower features (B, T, 84) -> temporal transformer -> logit (B, 1)

    Unlike the linear baseline which mean-pools (destroying temporal info),
    this model processes the sequence of bandpower snapshots to capture
    dynamics like:
      - mu desynchronization at movement onset
      - beta rebound after movement
      - temporal transitions between open/closed states

    Small and regularized to avoid overfitting on limited data.
    """

    def __init__(
        self,
        num_features: int = 84,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,  # small FFN to limit capacity
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Weighted pooling: learn which timesteps matter
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, 84)
        Returns:
            logits: (B, 1)
        """
        x = self.input_proj(features)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, T, d_model)
        x = self.norm(x)

        # Attention-weighted pooling across time
        attn_weights = self.attn_pool(x).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (B, T)
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        return self.head(pooled)  # (B, 1)
