import torch
import torch.nn as nn


class VQVAEDecoder(nn.Module):
    def __init__(self, output_dim: int, embedding_dim: int):
        """
        The VQ-VAE decoder will take in an embedding of size
        (B, T, embedding_dim) and will output a sequence of
        appendage vectors of size (B, T, 12). It will pass through
        multiple linear and 1 dimensional convolutional layers.
        """

        super().__init__()

        self.conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm1d(embedding_dim)
        
        self.conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm1d(embedding_dim)
        
        self.conv3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm1d(embedding_dim)
        
        self.conv4 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_4 = nn.BatchNorm1d(embedding_dim)
        
        self.conv5 = nn.Conv1d(embedding_dim, output_dim, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        The forward pass of the VQ-VAE decoder layer.
        """
        
        x = x.transpose(-1, -2)

        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn_3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn_4(x)
        x = self.relu(x)

        x = self.conv5(x)

        x = x.transpose(-1, -2)

        return x
