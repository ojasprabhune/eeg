import torch
import torch.nn as nn

from .transformer import Decoder 

class VQVAEDecoder(nn.Module):
    def __init__(self, output_dim: int, embedding_dim: int):
        """
        The VQ-VAE decoder will take in an embedding of size
        (B, T, embedding_dim) and will output a sequence of
        appendage vectors of size (B, T, 12).
        """

        super().__init__()

        self.decoder = Decoder(
            num_layers=4,
            num_heads=4,
            embedding_dim=embedding_dim,
            ffn_hidden_dim=embedding_dim,
            qk_length=64,
            value_length=64,
            max_length=2048,
            dropout=0.1,
        )

        self.linear1 = nn.Linear(embedding_dim, embedding_dim)

        self.leaky_relu = nn.LeakyReLU()


    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        The forward pass of the VQ-VAE decoder layer.
        """

        return x
