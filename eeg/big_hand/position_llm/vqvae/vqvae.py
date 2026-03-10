import torch
import torch.nn as nn

from .vqvae_encoder import VQVAEEncoder
from .vqvae_decoder import VQVAEDecoder


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 12,
        codebook_size: int = 512,
        embedding_dim: int = 64,
        decay: float = 0.99
    ) -> None:

        super().__init__()

        self.encoder = VQVAEEncoder(input_dim=input_dim, embedding_dim=embedding_dim, codebook_size=codebook_size, decay=decay)
        self.decoder = VQVAEDecoder(output_dim=input_dim, embedding_dim=embedding_dim)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VQ-VAE model forward: encode -> quantize -> decode.
        It returns:
        1. x: reconstructred input (T, 12)
        2. z_e: encoder output (continuous embeddings - T, C)
        3. z_q: quantized embeddings from codebook (T,)
        """

        z_e, z_q = self.encoder(x, return_toks=False)
        x_recon = self.decoder(z_q)
    
        return x_recon, z_e, z_q
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input sequence used to get the VQ-VAE tokens or the TOKEN IDs or INDICES
        of the quantized embeddings from the codebook.

        The model takes in an input sequence and encodes it into discrete latent
        tokens (T,).
        """

        q_token_ids = self.encoder(x, return_toks=True)

        return q_token_ids


    def decode(self, toks: torch.Tensor) -> torch.Tensor:
        """
        Use VQ-VAE tokens to return original data.

        The model decodes the discrete latent tokens back into an output
        sequence (T, 12).
        """
        z_q = self.encoder.codebook(toks)
        x = self.decoder(z_q)

        return x