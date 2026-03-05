import torch
import torch.nn as nn

from .transformer import Encoder, Decoder

class EEGLLM(nn.Module):
    """
    Big model to train on EEG and appendate values.
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        pass

    def forward(self):
        pass