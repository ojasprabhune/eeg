import torch
import torch.nn as nn

from .encoder import PositionalEncoding

from .attention import MultiHeadAttention, FeedForwardNN


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float = 0.1,
    ):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass

        self.MHA = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.MHA_CA = MultiHeadAttention(
            num_heads, embedding_dim, qk_length, value_length
        )
        self.FFN = FeedForwardNN(embedding_dim, ffn_hidden_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, enc_x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """

        # masked self attention
        residual_x = x
        x = self.MHA(x, x, x)  # for self attention, QKV are copies
        x = self.dropout(x)
        x += residual_x
        x = self.norm1(x)

        Q, K, V = x, enc_x, enc_x  # QKV are different

        # multi head cross attention
        residual_x = x
        x = self.MHA(Q, K, V)
        x = self.dropout(x)
        x += residual_x
        x = self.norm2(x)

        # feed forward network
        residual_x = x
        x = self.FFN(x)
        x = self.dropout(x)
        x += residual_x
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
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
        dropout: float = 0.1,
    ):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        #
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer decoder.

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
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
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def make_mask(self, x: torch.Tensor) -> torch.Tensor:
        # dictionary of input embeddings
        """
        Create a mask to prevent attention to future tokens.
        """

        B, T, C = x.size()
        ones = torch.ones((1, T, T))
        out = torch.tril(ones, 1)

        return out

    def forward(self, x: torch.Tensor, enc_x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """

        sequence_embedding = self.embedding(x)  # (B, T, C)
        decode_mask = self.make_mask(sequence_embedding)
        x = self.positional_encoding(sequence_embedding)
        x = self.dropout(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_x, decode_mask)
        x = self.linear(x)

        return x # (B, T, vocab_size)
