import torch
import torch.nn as nn


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

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_projection = nn.Linear(num_inputs_classes, embedding_dim)
        self.relu = nn.ReLU()

    def make_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a mask to prevent attention to future tokens.
        """

        B, T, C = x.size()
        ones = torch.ones((1, T, T))
        out = torch.tril(ones, 1)

        return out

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ """

        src = self.linear_projection(src)  # (B, T, C) -> (B, T, embedding_dim)
        src = self.relu(src)

        tgt = self.decoder_embedding(tgt)

        pred = self.transformer(
            src,
            tgt,
            src_mask=self.make_mask(src),
            tgt_mask=self.make_mask(tgt),
            src_key_padding_mask=mask,
            tgt_key_padding_mask=mask,
            memory_key_padding_mask=mask
        )

        return pred
