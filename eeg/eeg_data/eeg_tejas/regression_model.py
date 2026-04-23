"""
Regression models for EEG bandpower -> hand keypoints.

Includes CorrMSELoss: combined MSE + correlation loss that rewards
both correct scale AND trajectory tracking.
"""

import math
import torch
import torch.nn as nn


class CorrMSELoss(nn.Module):
    """
    Combined loss: MSE + lambda * (1 - mean_correlation).

    MSE handles scale/offset, correlation handles trajectory tracking.
    Without the correlation term, the model collapses to predicting
    the window mean (low MSE, zero correlation).
    """

    def __init__(self, corr_weight: float = 1.0):
        super().__init__()
        self.corr_weight = corr_weight
        self.mse = nn.MSELoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred: (B, T, D) or (B, D)
            target: same shape as pred
        Returns:
            loss: scalar
            components: dict with mse, corr_loss, mean_corr for logging
        """
        mse_loss = self.mse(pred, target)

        # Flatten to (N, D) if needed
        if pred.dim() == 3:
            B, T, D = pred.shape
            pred_flat = pred.reshape(B * T, D)
            target_flat = target.reshape(B * T, D)
        else:
            pred_flat = pred
            target_flat = target

        # Per-dimension correlation across the batch*time axis
        # Center
        pred_c = pred_flat - pred_flat.mean(dim=0, keepdim=True)
        target_c = target_flat - target_flat.mean(dim=0, keepdim=True)

        # Correlation per dimension
        num = (pred_c * target_c).sum(dim=0)  # (D,)
        den = torch.sqrt((pred_c**2).sum(dim=0) * (target_c**2).sum(dim=0)) + 1e-8
        corr = num / den  # (D,)

        mean_corr = corr.mean()
        corr_loss = 1.0 - mean_corr

        total = mse_loss + self.corr_weight * corr_loss

        return total, {
            "mse": mse_loss.item(),
            "corr_loss": corr_loss.item(),
            "mean_corr": mean_corr.item(),
        }


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


class EEGRegressionBaseline(nn.Module):
    """
    Bandpower (B, T, 84) -> mean pool -> (B, 12)
    """

    def __init__(
        self,
        num_features: int = 84,
        app_dim: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, app_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1)
        return self.head(pooled)


class EEGRegressionTemporal(nn.Module):
    """
    Bandpower (B, T, 84) -> transformer -> (B, T, 12)

    Per-timestep prediction with a residual connection from the
    mean prediction — lets the model learn dynamics as a delta
    on top of a good static estimate.
    """

    def __init__(
        self,
        num_features: int = 84,
        app_dim: int = 12,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.app_dim = app_dim

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Static head: predicts mean pose from pooled features
        self.static_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, app_dim),
        )

        # Dynamic head: predicts per-timestep delta from mean
        self.dynamic_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, app_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, 84)
        Returns:
            pred: (B, T, 12)
        """
        x = self.input_proj(features)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, T, d_model)
        x = self.norm(x)

        # Static: mean pose from pooled representation
        pooled = x.mean(dim=1)  # (B, d_model)
        static_pred = self.static_head(pooled)  # (B, app_dim)
        static_pred = static_pred.unsqueeze(1)  # (B, 1, app_dim)

        # Dynamic: per-timestep deviation
        dynamic_pred = self.dynamic_head(x)  # (B, T, app_dim)

        # Final = static base + dynamic delta
        return static_pred + dynamic_pred  # (B, T, app_dim)
