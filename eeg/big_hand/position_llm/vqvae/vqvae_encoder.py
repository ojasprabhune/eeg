import torch
import torch.nn as nn

class VQVAEEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, codebook_size: int, decay: float, epsilon: float = 1e-5):
        """
        The VQ-VAE encoder takes in a sequence of appendage
        vectors of size (T, 12) and passes it through a
        series of 1 dimensional convolutional layers to output a
        an embedding of size (T, embedding_dim).
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
        
        self.relu = nn.ReLU()

        self.codebook = nn.Embedding(num_embeddings=codebook_size,
                                     embedding_dim=embedding_dim)
        self.codebook.weight.data.normal_()
        self.register_buffer('_ema_cluster_size', torch.zeros(codebook_size))
        self._ema_w = nn.Parameter(torch.Tensor(codebook_size, embedding_dim))
        self._ema_w.data.normal()


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

        z_e = x.transpose(-1, -2)

        dist = torch.cdist(z_e, self.codebook.weight, p=2.0)

        q_token_ids = torch.argmin(dist, dim=-1)

        if return_toks:
            return q_token_ids # (B, T)

        z_q = self.codebook.weight[q_token_ids]

        z_q = z_e + (z_q - z_e).detach() # (B, T, C)

        return z_e, z_q
