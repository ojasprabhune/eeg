import math

import torch
import torch.nn as nn

from braindecode.models import Labram 

class LabramModel(nn.Module):
    def __init__(self,
                 embedding_dim: int = 64,
                 vocab_size: int = 3,
                 num_channels: int = 64,
                 num_times: int = 800,
                 num_outputs: int = 0) -> None:
        """
        EEGLLM with LaBraM backbone and neural network head.
        """

        super().__init__()

        self.model = Labram(n_chans=num_channels,
                       n_times=num_times,
                       n_outputs=num_outputs)
        self.pretrained_state = torch.load("scripts/models/labram_nc64_nt800_backbone.pt", map_location="cpu")
        self.model.load_state_dict(self.pretrained_state)       

        # freeze labram parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # |x| = C * floor(t / w)
        time_window = 200
        num_patches = num_channels * math.floor(num_times / time_window)

        self.linear = nn.Linear(time_window, time_window)

        self.linear1 = nn.Linear(num_patches, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.linear3 = nn.Linear(embedding_dim, embedding_dim)
        self.linear4 = nn.Linear(embedding_dim, vocab_size)

        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor):
        """
        Input is shape (B, C, T)
        """

        x = self.model.forward_features(x, return_patch_tokens=True) # B, num_patches, T

        x = self.linear(x)
        x = self.relu(x)

        x = x.transpose(-1, -2) # B, T, num_patches

        x = self.linear1(x) # B, T, C
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x) # B, T, vocab_size
        
        return x
