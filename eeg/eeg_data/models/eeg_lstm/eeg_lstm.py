import torch
import torch.nn as nn

class EEGLSTM(nn.Module):
    """
    EEG LSTM model for processing EEG data.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        num_layers: int = 2,
        num_channels: int = 14,
        embedding_dim: int = 128,
        kernel_size_temporal: int = 7,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, vocab_size)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EEG LSTM model which consists of:
        1. An input: EEG data of shape (B, T, C)
        1. Pass through temporal convolution
        1. Pass through the LSTM for output of shape (B, T, 2 * C)
        1. Linear layer projection to shape (B, T, vocab_size)
        """

        x = x.transpose(1, 2)     # (B, C, T)
        x = self.temporal_conv(x) # (B, 64, T)
        x = x.transpose(1, 2)     # (B, T, 64)

        x, (h_n, c_n) = self.lstm(x)
        x = self.head(x)

        return x