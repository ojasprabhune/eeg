import torch
import torch.nn as nn

from .transformer import PositionalEncoding


class TemporalModel(nn.Module):
    """
    Bandpower features (B, T, 84) -> temporal transformer -> logit (B, 4)

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
        vocab_size: int = 4,
    ) -> None:
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

        # weighted pooling: learn which timesteps matter
        self.attn_pool = nn.Linear(d_model, vocab_size)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input projection: "what is happening at each moment?"
        Positional encoding: "where are we in the sequence?"
        Transformer encoder: "how do moments relate to each other?"
        Attention pooling: "which moments matter most overall?"

        Args:
            features: (B, T, 84)
        Returns:
            logits: (B, vocab_size)
        """
        x = self.input_proj(features)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, T, d_model)
        x = self.norm(x)

        # attention-weighted pooling across time
        attn_weights = self.attn_pool(x).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (B, T)
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        return self.head(pooled)  # (B, vocab_size)
