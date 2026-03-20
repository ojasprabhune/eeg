import torch
import torch.nn as nn

class EEGLSTM(nn.Module):
    """
    EEG LSTM model for processing EEG data.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        num_layers: int = 8,
        num_channels: int = 14,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.linear = nn.Linear(2 * embedding_dim, vocab_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EEG LSTM model which consists of:
        1. An input: EEG data of shape (B, T, C)
        2. Pass through the LSTM for output of shape (B, T, 2 * C)
        3. Linear layer projection to shape (B, T, vocab_size)
        """

        x = self.lstm(x)
        x, (h_n, c_n) = self.linear(x)

        return x