import torch
import torch.nn as nn

from .transformer import Encoder 

class VQVAEDecoder(nn.Module):
    def __init__(self, output_dim: int, embedding_dim: int):
        """
        The VQ-VAE decoder will take in an embedding of size
        (B, T, embedding_dim) and will output a sequence of
        appendage vectors of size (B, T, 12).
        """

        super().__init__()

        self.encoder = Encoder(
            num_layers=4,
            num_heads=4,
            embedding_dim=embedding_dim,
            ffn_hidden_dim=embedding_dim,
            qk_length=64,
            value_length=64,
            max_length=2048,
            dropout=0.1,
        )

        self.linear = nn.Linear(embedding_dim, output_dim)


    def forward(self, x: torch.Tensor):
        """
        The forward pass of the VQ-VAE decoder layer.
        """

        x = self.encoder(x) # B, T, C
        x = self.linear(x) # B, T, 12

        return x
