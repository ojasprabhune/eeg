import torch
import torch.nn as nn

class EEG_CNN(nn.Module):
    def __init__(self,
                 num_features: int = 40,
                 kernel_size_temporal: int = 30,
                 kernel_size_spatial: int = 14,
                 kernel_size_avg_pool: int = 15,
                 ffn_embedding_dim: int = 80,
                 seq_len: int = 900,
                 vocab_size: int = 3) -> None:
        """
        EEG CNN to perform classification.
        """

        super().__init__()

        # same explicitly tells Pytorch to pad the input so that the output has the same shape as the input
        self.temporal_conv = nn.Conv2d(in_channels=1,
                                       out_channels=num_features,
                                       kernel_size=(1, kernel_size_temporal),
                                       padding="same")

        # valid explicitly tells Pytorch to not pad the input, so the channels are squeezed together
        self.spatial_conv = nn.Conv2d(in_channels=num_features,
                                       out_channels=num_features,
                                       kernel_size=(kernel_size_spatial, 1),
                                       padding="valid")

        self.batch_norm1 = nn.BatchNorm2d(num_features)
        self.batch_norm2 = nn.BatchNorm2d(num_features)

        self.elu = nn.ELU()

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, kernel_size_avg_pool))

        flattened_features = num_features * 1 * (seq_len // kernel_size_avg_pool) # 40 * 1 * 60 = 2400

        self.fc1 = nn.Linear(flattened_features, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, vocab_size)

        self.softmax = nn.Softmax(dim=1) # softmax across the vocab dimension


    def forward(self, x: torch.Tensor):
        """
        Input is shape (B, C, T) or (N, C, T).
        """

        x = x.unsqueeze(1) # (N, C, H, W) where C is 1 and H is num_channels

        x = self.temporal_conv(x) # (B, num_features, H, W)
        x = self.batch_norm1(x)
        x = self.elu(x)

        x = self.spatial_conv(x) # (B, num_features, 1, W)
        x = self.batch_norm2(x)
        x = self.elu(x)

        x = self.avg_pool(x) # (B, C, 1, W / kernel_size_avg_pool)
        x = torch.flatten(x, start_dim=1) # (B, num_flattened_features)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x) # (B, vocab_size)

        return x
