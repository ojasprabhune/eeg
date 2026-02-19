import torch
import torch.nn as nn

class EEG_CNN(nn.Module):
    def __init__(self,
                 num_features: int = 40,
                 kernel_size_temporal: int = 30,
                 kernel_size_spatial: int = 14,
                 kernel_size_avg_pool: int = 15
                 ffn_embedding_dim: int = 80,
                 vocab_size: int = 3) -> None:
        """
        EEG CNN to perform classification.
        """

        super().__init__()

        self.temporal_conv = nn.Conv2d(in_channels=1,
                                       out_channels=num_features,
                                       kernel_size=(1, kernel_size_temporal),
                                       padding="same")

        self.spatial_conv = nn.Conv2d(in_channels=num_features,
                                       out_channels=num_features,
                                       kernel_size=(kernel_size_spatial, 1),
                                       padding="same")

        self.batch_norm = nn.BatchNorm2d(num_features)

        self.elu = nn.ELU()

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, kernel_size_avg_pool))

        self.ffn_1 = nn.Linear(ffn_embedding_dim)


    def forward(self, x: torch.Tensor):
        """
        Input is shape (B, C, T) or (N, C, T).
        """

        x = x.unsqueeze(1) # (N, C, H, W) where C is 1 and H is num_channels

        x = self.temporal_conv(x) # (32, 40, 14, 900)
        x = self.batch_norm(x) # (32, 40, 14, 900)
        x = self.elu(x) # (32, 40, 14, 900)

        x = self.spatial_conv(x) # (32, 40, 1, 900)
        x = self.batch_norm(x) # (32, 40, 1, 900)
        x = self.elu(x) # (32, 40, 1, 900)

        x = self.avg_pool(x) # (32, 40, 1, 60)
        
        return x
