import torch
import torch.nn as nn

from .transformer.attention import FeedForwardNN, MultiHeadAttention
from .transformer.encoder import PositionalEncoding


class PositionLLMLayer(nn.Module):
    """
    Essentially equivalent to a DecoderLayer without cross-attention
    since we are just doing next-token prediction on k-means regions.
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # instance attributes

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # layers

        self.dropout = nn.Dropout(p=dropout)
        self.MHA = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.FFN = FeedForwardNN(embedding_dim, ffn_hidden_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class PositionLLM(nn.Module):
    """
    Essentially a TransformerDecoder without cross-attention.
    """

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
        super().__init__()

        # instance attributes

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # layers

        # dictionary of token embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positionllm_layers = nn.ModuleList(
            [
                PositionLLMLayer(
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
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, dropout, max_length
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_embedding = self.embedding(x)  # (B, T, C)
        # print(f"emb shape: {sequence_embedding.shape}")
        x = self.positional_encoding(sequence_embedding)  # add positional information
        # print(f"posenc shape: {x.shape}")
        x = self.dropout(x)

        # process data on layers
        for positionllm_layer in self.positionllm_layers:
            x = positionllm_layer(x)

        x = self.linear(x)

        return x
