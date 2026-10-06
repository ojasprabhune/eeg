import torch
from torch import nn

from .transformer import PositionalEncoding


class GestureModel(nn.Module):
    """
    A transformer encoder and decoder model that takes in EEG features of
    shape (B, T, C) of an epoch and outputs a probability distribution over
    gesture classes for that ENTIRE epoch (B, num_classes).

    The decoder is unusual in that it uses a single learned query that
    represents the predicted class for the epoch. It cross-attends into the
    encoder's memory and has a pooled classification vector.
    """

    def __init__(
        self,
        num_features: int = 84,
        num_classes: int = 6,
        num_layers: int = 4,
        decoder_num_layers: int = 4,
        num_heads: int = 4,
        embedding_dim: int = 64,
        ffn_hidden_dim: int = 64,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.pos_enc = PositionalEncoding(embedding_dim, dropout=encoder_dropout)
        self.linear_projection = nn.Linear(num_features, embedding_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=encoder_dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # a single learnable vector standing in for the decoder's "target
        # sequence". it cross-attends onto the encoded epoch and comes
        # back out holding a pooled summary of it
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=decoder_dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=decoder_num_layers,
        )

        self.class_projection = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, src: torch.Tensor, src_pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Input: EEG features for one epoch (B, T, C) where C is initially
        the number of channels, and a bool mask (B, T) indicating padded
        positions.

        Output: logits (B, num_classes) for the entire epoch.
        """
        B = src.size(0)

        src = self.linear_projection(src)  # (B, T, C)
        src = self.pos_enc(src)

        memory: torch.Tensor = self.encoder(
            src, src_key_padding_mask=src_pad_mask
        )  # (B, T, C)

        # expand so that the same query is used for every batch
        query = self.query.expand(B, -1, -1)  # (B, 1, C)

        pooled = self.decoder(
            query,
            memory,
            memory_key_padding_mask=src_pad_mask,
        )  # (B, 1, C)

        logits: torch.Tensor = self.class_projection(
            pooled.squeeze(1)
        )  # (B, num_classes)

        return logits
