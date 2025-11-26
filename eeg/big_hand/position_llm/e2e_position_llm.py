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

        # expected: (T, 50)
        region_logits = self.position_llm(region_tokens)

        compact_delta_logits = self.linear(
            region_logits)  # expected: (T, 63 * 210)

        # expected: (T, 63, 210)
        delta_logits = torch.reshape(compact_delta_logits, (B, T, 63, 210))

        return region_logits, delta_logits
