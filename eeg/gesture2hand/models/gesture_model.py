import torch
import torch.nn as nn

from .transformer import PositionalEncoding


class GestureModel(nn.Module):
    """
    EEG features for one epoch (B, T, C) -> transformer encoder/decoder ->
    a probability distribution over gesture classes for that ENTIRE epoch
    (B, num_classes).

    Unlike eeg/language_model/language_model.py, this isn't sequence-to-
    sequence — there's no autoregressive target to generate, since each
    epoch is already exactly one letter. The decoder is still doing real
    work though: instead of decoding a token sequence, a single learnable
    query cross-attends onto the encoder's memory and comes back out
    holding a pooled classification vector, the same role <SOS> plays at
    the start of the language model's decoder, but taking only one step
    since there's only one label per epoch. This keeps the same
    TransformerEncoder + TransformerDecoder shape as the language model
    while fitting a single-label task.
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
        # sequence" - it cross-attends onto the encoded epoch and comes
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
        Args:
            src: (B, T, C) EEG features for one epoch (one letter).
            src_pad_mask: (B, T) bool mask, True at padded positions.
        Returns:
            logits: (B, num_classes)
        """
        B = src.size(0)

        src = self.linear_projection(src)  # (B, T, C) -> (B, T, embedding_dim)
        src = self.pos_enc(src)

        memory: torch.Tensor = self.encoder(
            src, src_key_padding_mask=src_pad_mask
        )  # (B, T, embedding_dim)

        query = self.query.expand(B, -1, -1)  # (B, 1, embedding_dim)

        pooled = self.decoder(
            query,
            memory,
            memory_key_padding_mask=src_pad_mask,
        )  # (B, 1, embedding_dim)

        logits: torch.Tensor = self.class_projection(
            pooled.squeeze(1)
        )  # (B, num_classes)

        return logits
