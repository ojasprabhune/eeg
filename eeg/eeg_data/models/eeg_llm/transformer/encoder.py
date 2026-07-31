import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, FeedForwardNN


class PositionalEncoding(nn.Module):
    """
    The PositionalEncoding layer will take in an input tensor
    of shape (B, T, C) and will output a tensor of the same
    shape, but with positional encodings added to the input.

    We provide you with the full implementation for this
    homework.

    Based on:
        https://web.archive.org/web/20230315052215/https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize the PositionalEncoding layer."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape (B, T, C)
        """
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)] # type: ignore
        x = self.dropout(x)
        return x.transpose(0, 1)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float,
    ):
        """
        Each encoder layer will take in an embedding of
        shape (B, T, C) and will output an encoded representation
        of the same shape.

        The encoder layer will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding!
        """
        super().__init__()

        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass

        self.dropout = nn.Dropout(
            p=dropout
        )  # weights spread out using dropout <- search online

        self.MHA = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.FFN = FeedForwardNN(embedding_dim, ffn_hidden_dim)

        # function: across all features, subtract mean and divide by standard deviation to get variance of 1
        # purpose: standardize gradients (similar to sqrt(qk)) - better training
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the EncoderLayer.
        """

        Q, K, V = x, x, x  # copies of input embedding

        # multi head attention
        residual_x = x
        x = self.MHA(Q, K, V)
        x = self.dropout(x)
        x += residual_x
        x = self.norm1(x)

        # feed forward network
        residual_x = x
        x = self.FFN(x)
        x = self.dropout(x)
        x += residual_x
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,
        dropout: float,
    ) -> None:
        """
        The EEG encoder will take in EEG data of shape (B, T, C) and will
        output an encoded representation of shape (B, T, C). The forward pass of
        the encoder consists of:
        1. Positional encoding
        1. Drop out
        1. Then passing this through the encoder several times, outputting a hidden representation
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads,
                    embedding_dim,
                    ffn_hidden_dim,
                    qk_length,
                    value_length,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim, dropout, max_length
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Encoder.
        """

        x = self.positional_encoding(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x
