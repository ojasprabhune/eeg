import torch
import torch.nn as nn


class LanguageModel(nn.Module):
    """ """

    def __init__(
        self,
        vocab_size: int = 29,
        batch_size: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        num_inputs_classes: int = 4,
        embedding_dim: int = 64,
        ffn_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.num_heads = num_heads

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_projection = nn.Linear(num_inputs_classes, embedding_dim)
        self.relu = nn.ReLU()
        self.vocab_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ """

        src = self.linear_projection(src)  # (B, T, C) -> (B, T, embedding_dim)
        src = self.relu(src)

        tgt = self.decoder_embedding(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        )

        pred = self.transformer(
            src,
            tgt,
            src_mask=None,  # causal
            tgt_mask=tgt_mask,  # causal
            src_key_padding_mask=src_pad_mask,  # padding
            tgt_key_padding_mask=tgt_pad_mask,  # padding
            memory_key_padding_mask=src_pad_mask,  # padding
        )

        x = self.vocab_projection(pred)

        return x
