import torch
import torch.nn as nn

from .transformer import Encoder, Decoder

class EEGLLM(nn.Module):
    """
    Big model to train on EEG and appendate values.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        embedding_dim: int = 64,
        ffn_hidden_dim: int = 64,
        qk_length: int = 64,
        value_length: int = 64,
        max_length: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x