import torch
import torch.nn as nn

from .vqvae_encoder import VQVAEEncoder
from .vqvae_decoder import VQVAEDecoder


class VQVAE(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, embedding_dim: int, decay: float = 0.99):
        super().__init__()

        self.encoder = VQVAEEncoder(input_dim=input_dim, embedding_dim=embedding_dim, codebook_size=codebook_size, decay=decay)
        self.decoder = VQVAEDecoder(output_dim=input_dim, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor, return_toks: bool = False):

        if return_toks:
            q_token_ids = self.encoder(x, return_toks=True)

            return q_token_ids
        else:
            z_e, z_q = self.encoder(x, return_toks=False)
            x = self.decoder(z_q)
        
            return x, z_e, z_q

    def decode(self, toks: torch.Tensor):

        z_q = self.encoder.codebook(toks)
        x = self.decoder(z_q)

        return x