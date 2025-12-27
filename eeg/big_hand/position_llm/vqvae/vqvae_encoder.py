import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Encoder

class VQVAEEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, codebook_size: int, decay: float, epsilon: float = 1e-5):
        """
        The VQ-VAE encoder takes in a sequence of appendage
        vectors of size (B, T, 12) and outputs a
        an embedding of size (B, T, embedding_dim).
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
        
        self.linear = nn.Linear(input_dim, embedding_dim)

        self.leaky_relu = nn.LeakyReLU()

        self.codebook = nn.Embedding(num_embeddings=codebook_size,
                                     embedding_dim=embedding_dim)
        self.codebook.weight.data.normal_()
        self.register_buffer('_ema_cluster_size', torch.zeros(codebook_size))
        self._ema_w = nn.Parameter(torch.Tensor(codebook_size, embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon


    def forward(self, x: torch.Tensor, return_toks: bool = False):
        """
        The forward pass of the VQ-VAE encoder layer.

        Input size: (B, T, 12)
        """

        x = self.linear(x) # B, T, C
        z_e = self.encoder(x) # B, T, C

        dist = torch.cdist(z_e, self.codebook.weight, p=2.0)

        q_token_ids = torch.argmin(dist, dim=-1)

        if return_toks:
            return q_token_ids # (B, T)

        # EMA update
        z_q = self.codebook.weight[q_token_ids]

        q_token_ids_flat = q_token_ids.view(-1)
        z_e_flat = z_e.reshape(-1, z_e.size(-1))
        encodings = F.one_hot(q_token_ids_flat, num_classes=self.codebook.num_embeddings).type(z_e.dtype)
        n_i = torch.sum(encodings, dim=0)
        sum_current = encodings.T @ z_e_flat


        self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                    (1 - self._decay) * n_i

        # Laplace smoothing of the cluster size
        n = torch.sum(self._ema_cluster_size)
        smoothed_cluster_size = (
            (self._ema_cluster_size + self._epsilon) 
            / (n + self.codebook.num_embeddings * self._epsilon) 
            * n
        )

        self._ema_w = nn.Parameter(self._ema_w * self._decay + \
                        (1 - self._decay) * sum_current)

        normalized_weight = self._ema_w / smoothed_cluster_size.unsqueeze(1)
        self.codebook.weight.data.copy_(normalized_weight)

        z_q = z_e + (z_q - z_e).detach() # (B, T, C)

        return z_e, z_q

if __name__ == "__main__":
    B, T, C = 2, 16, 12
    x = torch.randn(B, T, C)

    model = VQVAEEncoder(input_dim=C, embedding_dim=64, codebook_size=512, decay=0.99)

    z_e, z_q = model(x)

    print(z_e.shape)  # (B, T, embedding_dim)
    print(z_q.shape)  # (B, T, embedding_dim)
