import torch
import torch.nn as nn

from .position_llm import PositionLLM


class E2EPositionLLM(nn.Module):
    def __init__(self,
                 vocab_size=512,
                 num_layers=4,
                 num_heads=4,
                 embedding_dim=64,
                 ffn_hidden_dim=64,
                 qk_length=64,
                 value_length=64,
                 max_length=2048,
                 dropout=0.1,
                 ):
        """
        End to end position LLM.
        """
        super().__init__()

        self.position_llm = PositionLLM(
            # vocab size is amount of regions
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            qk_length=qk_length,
            value_length=value_length,
            max_length=max_length,
            dropout=dropout,
        )

        # predict appendage vectors based on regions
        self.linear = nn.Linear(50, 12)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Needs to have shape (B, T)
        """

        region_tokens = x
        B, T = x.shape

        # expected: (B, T, 50)
        region_logits = self.position_llm(region_tokens)

        appendage_values = self.linear(region_logits)  # expected: (B, T, 12)

        return region_logits, appendage_values
