import torch
import torch.nn as nn


class VQVAEDecoder(nn.Module):
    def __init__(self, output_dim: int, embedding_dim: int):
        """
        The VQ-VAE decoder will take in an embedding of size
        (T, embedding_dim) and will output a sequence of
        appendage vectors of size (B, T, 12). It will pass through
        multiple linear and 1 dimensional convolutional layers.
        """

        super().__init__()

        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear2 = nn.Linear(in_features=embedding_dim, out_features=output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        The forward pass of the VQ-VAE decoder layer.
        """
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x