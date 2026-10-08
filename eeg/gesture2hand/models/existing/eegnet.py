import torch
from torch import nn


class EEGNet(nn.Module):
    """
    https://github.com/vlawhern/arl-eegmodels
    https://arxiv.org/pdf/1611.08024

    PyTorch implementation of EEGNet, adapted to this project.
    """

    def __init__(
        self,
        vocab_size: int = 4,
        num_channels: int = 64,
        num_samples: int = 160,
        dropout: float = 0.5,
        kern_length: int = 64,
        f1: int = 8,
        f2: int = 16,
        d: int = 2,
    ) -> None:

        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=f1,
            kernel_size=(1, kern_length),
            padding="same",
            bias=False,
        )

        self.batch_norm1 = nn.BatchNorm2d(f1)

        self.depthwise_conv2d = nn.Conv2d(
            in_channels=f1,
            out_channels=d * f1,
            kernel_size=(num_channels, 1),
            groups=f1,
            bias=False,
        )

        self.batch_norm2 = nn.BatchNorm2d(d * f1)

        self.elu = nn.ELU()

        self.avg_pool1 = nn.AvgPool2d(
            kernel_size=(1, 4),
        )

        self.dropout = nn.Dropout2d(dropout)

        self.separable_conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=f1 * d,
                out_channels=f1 * d,
                kernel_size=(1, 16),
                padding="same",
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(f1 * d, f2, kernel_size=1, bias=False),
        )

        self.batch_norm3 = nn.BatchNorm2d(f2)

        self.avg_pool2 = nn.AvgPool2d(
            kernel_size=(1, 8),
        )

        self.vocab_projection = nn.Linear(
            in_features=f2 * int(num_samples / 32),
            out_features=vocab_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes in a tensor of shape (batch_size, 1, num_channels, num_samples)
        and returns a tensor of shape (batch_size, vocab_size).

        Simply, this model is special because it uses depthwise separable
        convolutions to process EEG data.
        """

        print(x.shape)

        # --- block 1 ---
        x = self.conv2d(x)
        print(x.shape)
        x = self.batch_norm1(x)
        print(x.shape)
        x = self.depthwise_conv2d(x)
        print(x.shape)
        x = self.batch_norm2(x)
        print(x.shape)
        x = self.elu(x)
        print(x.shape)
        x = self.avg_pool1(x)
        print(x.shape)
        x = self.dropout(x)

        print(x.shape)

        # --- block 2 ---
        x = self.separable_conv2d(x)
        x = self.batch_norm3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)

        print(x.shape)

        x = x.flatten(start_dim=1)

        print(x.shape)
        quit()

        # --- classifier ---
        x = self.vocab_projection(x)
        return x
