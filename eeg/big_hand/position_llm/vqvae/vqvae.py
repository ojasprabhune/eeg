import torch
import torch.nn as nn

from .vqvae_encoder import VQVAEEncoder
from .vqvae_decoder import VQVAEDecoder


class VQVAE(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, embedding_dim: int, decay: float = 0.99):
        super().__init__()

        self.encoder = VQVAEEncoder(input_dim=input_dim, embedding_dim=embedding_dim, codebook_size=codebook_size, decay=decay)
        self.decoder = VQVAEDecoder(output_dim=input_dim, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor):

        z_e, z_q = self.encoder(x)
        
        x = self.decoder(z_q)
        
        return x, z_e, z_q
