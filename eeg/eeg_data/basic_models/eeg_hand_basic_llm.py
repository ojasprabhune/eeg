import math

import torch
import torch.nn as nn

from .transformer.decoder import Decoder
from braindecode.models import Labram 


class EEGLLM(nn.Module):
    """
    EEGLLM that computes basic hand positions.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,

        num_channels: int,
        num_times: int,
        num_outputs: int,

        dropout: float,
    ) -> None:
        """

        """

        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length
        self.max_length = max_length

        self.num_channels = num_channels
        self.num_times = num_times
        self.num_outputs = num_outputs

        self.dropout = dropout

        # --- LaBraM ---
        self.labram = Labram(n_chans=num_channels, n_times=num_times, n_outputs=num_outputs)
        self.pretrained_state = torch.load("scripts/models/labram_nc64_nt800_backbone.pt", map_location="cpu")
        self.labram.load_state_dict(self.pretrained_state)       

        # freeze labram parameters
        for param in self.labram.parameters():
            param.requires_grad = False

        # |x| = C * floor(t / w)
        time_window = 200
        num_patches = num_channels * math.floor(num_times / time_window)

        self.embedding_dim_linear = nn.Linear(num_patches, embedding_dim)

        # --- Decoder ---
        self.decoder = Decoder(
            vocab_size = vocab_size,
            num_layers = num_layers,
            num_heads = num_heads,
            embedding_dim = embedding_dim,
            ffn_hidden_dim = ffn_hidden_dim,
            qk_length = qk_length,
            value_length = value_length,
            max_length = max_length 
            )


    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        EEGLLM takes in EEG data of size (B, C, T) and a target representation,
        where B is batch size, C is number of channels, and T is sequence
        length. Returns a probability distribution over hand positions
        (0, 1, 2) of size (B, T, vocab_size).
        """

        h = self.labram.forward_features(x, return_patch_tokens=True)   # h: (B, num_patches, T)
        h = h.transpose(-1, -2)                                         # h: (B, T, num_patches)
        h = self.embedding_dim_linear(h)                                # h: (B, T, C)
        x = self.decoder(target, h)                                     # x: (B, T, vocab_size)

        return x
