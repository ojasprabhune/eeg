import torch
import torch.nn as nn

from .position_llm import PositionLLM


class E2EPositionLLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.position_llm = PositionLLM(
            # vocab size is amount of regions
            vocab_size=50,
            num_layers=1,
            num_heads=1,
            embedding_dim=64,
            ffn_hidden_dim=64,
            qk_length=64,
            value_length=64,
            max_length=2048,
            dropout=0.1,
        )

        # predict appendage vectors based on regions
        self.linear = nn.Linear(50, 12)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Needs to have shape (B, T)"""
        region_tokens = x
        B, T = x.shape

        # expected: (B, T, 50)
        region_logits = self.position_llm(region_tokens)

        appendage_values = self.linear(region_logits)  # expected: (B, T, 12)

        return region_logits, appendage_values
