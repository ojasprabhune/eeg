import torch
import torch.nn as nn

from .transformer import Decoder 

class VQVAEDecoder(nn.Module):
    def __init__(self, output_dim: int, embedding_dim: int):
        """
        The VQ-VAE decoder will take in an embedding of size
        (B, T, embedding_dim) and will output a sequence of
        appendage vectors of size (B, T, 12). It will pass through
        multiple linear and 1 dimensional convolutional layers.
        """

        super().__init__()

        self.decoder = Decoder()

        self.rnn1 = nn.RNN(input_size=embedding_dim,
                          hidden_size=embedding_dim,
                          num_layers=4,
                          nonlinearity="tanh",
                          batch_first=True,
                          )

        self.rnn2 = nn.RNN(input_size=embedding_dim,
                          hidden_size=embedding_dim,
                          num_layers=4,
                          nonlinearity="tanh",
                          batch_first=True,
                          )

        self.conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm1d(embedding_dim)
        
        self.conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm1d(embedding_dim)
        
        self.conv3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm1d(embedding_dim)
        
        self.conv4 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_4 = nn.BatchNorm1d(embedding_dim)
        
        self.conv5 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_5 = nn.BatchNorm1d(embedding_dim)
        
        self.conv6 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_6 = nn.BatchNorm1d(embedding_dim)
        
        self.conv7 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn_7 = nn.BatchNorm1d(embedding_dim)
        
        self.conv8 = nn.Conv1d(embedding_dim, output_dim, kernel_size=3, stride=1, padding=1)

        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.linear3 = nn.Linear(embedding_dim, embedding_dim)
        self.linear4 = nn.Linear(embedding_dim, embedding_dim)
        self.linear5 = nn.Linear(embedding_dim, embedding_dim)
        self.linear6 = nn.Linear(embedding_dim, embedding_dim)
        self.linear7 = nn.Linear(embedding_dim, embedding_dim)
        self.linear8 = nn.Linear(embedding_dim, embedding_dim)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()


    def forward(self, x: torch.Tensor):
        """
        The forward pass of the VQ-VAE decoder layer.
        """

        x, hn = self.rnn1(x)
        x, hn = self.rnn2(x)
        
        x = x.transpose(-1, -2)

        start_x = x

        # x = x.transpose(-1, -2) # (B, T, C)
        # x = self.linear1(x)
        # x = x.transpose(-1, -2) # (B, C, T)
        x = self.leaky_relu(x)
        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = x + start_x

        start_x = x

        x = x.transpose(-1, -2) # (B, T, C)
        x = self.linear2(x)
        x = x.transpose(-1, -2) # (B, C, T)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = x + start_x

        start_x = x

        x = x.transpose(-1, -2) # (B, T, C)
        x = self.linear3(x)
        x = x.transpose(-1, -2) # (B, C, T)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn_3(x)
        x = self.relu(x)

        x = x + start_x

        start_x = x

        x = x.transpose(-1, -2) # (B, T, C)
        x = self.linear4(x)
        x = x.transpose(-1, -2) # (B, C, T)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn_4(x)
        x = self.relu(x)

        x = x + start_x

        x = x.transpose(-1, -2) # (B, T, C)
        x = self.linear5(x)
        x = x.transpose(-1, -2) # (B, C, T)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.bn_5(x)
        x = self.relu(x)

        x = x + start_x

        x = x.transpose(-1, -2) # (B, T, C)
        x = self.linear6(x)
        x = x.transpose(-1, -2) # (B, C, T)
        x = self.leaky_relu(x)
        x = self.conv6(x)
        x = self.bn_6(x)
        x = self.relu(x)

        x = x + start_x

        x = x.transpose(-1, -2) # (B, T, C)
        x = self.linear7(x)
        x = x.transpose(-1, -2) # (B, C, T)
        x = self.leaky_relu(x)
        x = self.conv7(x)
        x = self.bn_7(x)
        x = self.relu(x)

        x = x + start_x

        x = x.transpose(-1, -2) # (B, T, C)
        x = self.linear8(x)
        x = x.transpose(-1, -2) # (B, C, T)
        x = self.leaky_relu(x)
        x = self.conv8(x)

        x = x.transpose(-1, -2)

        return x
