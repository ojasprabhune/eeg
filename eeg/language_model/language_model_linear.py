import torch
import torch.nn as nn

from .transformer import PositionalEncoding


class LanguageModelLinear(nn.Module):
    """ """

    def __init__(
        self,
        input_seq_len: int,
        output_seq_len: int,
        vocab_size: int = 29,
        num_layers: int = 4,
        num_heads: int = 4,
        num_inputs_classes: int = 4,
        embedding_dim: int = 64,
        ffn_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.pos_enc = PositionalEncoding(embedding_dim, dropout=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
        )

        self.linear_projection = nn.Linear(num_inputs_classes, embedding_dim)
        self.seq_projection = nn.Linear(input_seq_len, output_seq_len)
        self.vocab_projection = nn.Linear(embedding_dim, vocab_size)

        self.ffn1 = nn.Linear(embedding_dim, embedding_dim)
        self.ffn2 = nn.Linear(embedding_dim, embedding_dim)
        self.ffn3 = nn.Linear(embedding_dim, embedding_dim)

        self.relu = nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        src_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        src = self.linear_projection(src)  # (B, T_in, C) -> (B, T_in, embedding_dim)
        src = self.pos_enc(src)

        latent: torch.Tensor = self.encoder(
            src, src_key_padding_mask=src_pad_mask
        )  # (B, T_in, C)

        latent = latent.transpose(1, 2)  # (B, T_in, C)
        latent = self.seq_projection(latent)  # (B, C, T_out)
        latent = self.relu(latent)
        latent = latent.transpose(1, 2)  # (B, T_out, C)

        x = self.ffn1(latent)
        x = self.relu(x)
        x = self.ffn2(latent)
        x = self.relu(x)
        x = self.ffn3(latent)
        x = self.relu(x)

        x = self.vocab_projection(x)

        return x
