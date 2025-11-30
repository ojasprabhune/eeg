import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAEEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, codebook_size: int, decay: float, epsilon: float = 1e-5):
        """
        The VQ-VAE encoder takes in a sequence of appendage
        vectors of size (B, T, 12) and passes it through a
        series of 1 dimensional convolutional layers to output a
        an embedding of size (B, T, embedding_dim).
        """

        super().__init__()

        self.conv_1 = nn.Conv1d(in_channels=input_dim,
                                out_channels=embedding_dim,
                                kernel_size=4,
                                stride=1,
                                padding=2
                                )
        self.bn_1 = nn.BatchNorm1d(embedding_dim)
                                
        self.conv_2 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=embedding_dim,
                                kernel_size=4,
                                stride=1,
                                padding=1
                                )
        self.bn_2 = nn.BatchNorm1d(embedding_dim)

        self.conv_3 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=embedding_dim,
                                kernel_size=4,
                                stride=1,
                                padding=2
                                )
        self.bn_3 = nn.BatchNorm1d(embedding_dim)

        self.conv_4 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=embedding_dim,
                                kernel_size=4,
                                stride=1,
                                padding=1
                                )
        self.bn_4 = nn.BatchNorm1d(embedding_dim)

        self.conv_5 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=embedding_dim,
                                kernel_size=4,
                                stride=1,
                                padding=2
                                )
        self.bn_5 = nn.BatchNorm1d(embedding_dim)

        self.conv_6 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=embedding_dim,
                                kernel_size=4,
                                stride=1,
                                padding=1
                                )
        
        self.relu = nn.ReLU()

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

        Input size: (B, T, C)
        """
        
        x = x.transpose(-1, -2)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.relu(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.relu(x)

        x = self.conv_6(x)

        z_e = x.transpose(-1, -2)

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