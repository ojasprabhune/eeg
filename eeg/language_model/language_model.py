import torch
import torch.nn as nn

from .transformer import Decoder, Encoder


class LanguageModel(nn.Module):
    """ """

    def __init__(
        self,
        vocab_size: int = 29,
        num_layers: int = 4,
        num_heads: int = 4,
        num_inputs_classes: int = 4,
        embedding_dim: int = 64,
        ffn_hidden_dim: int = 64,
        qk_length: int = 64,
        value_length: int = 64,
        max_length: int = 2048,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            qk_length=qk_length,
            value_length=value_length,
            max_length=max_length,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            qk_length=qk_length,
            value_length=value_length,
            max_length=max_length,
            dropout=dropout,
        )

        self.linear_projection = nn.Linear(num_inputs_classes, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ """

        x = self.linear_projection(x)  # (B, T, C) -> (B, T, embedding_dim)
        x = self.relu(x)

        x_enc = self.encoder(x)
        x_dec = self.decoder(x, x_enc)

        return x_dec
