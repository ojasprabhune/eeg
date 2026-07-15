import math

import torch
import torch.nn as nn

from .transformer import PositionalEncoding


class LanguageModel(nn.Module):
    """
    A language model that predicts the corrosponding sequence of letters given
    a sequence of probability distributions of gesture classes of ASL letters
    based on EEG data.
    """

    def __init__(
        self,
        vocab_size: int = 29,
        num_layers: int = 4,
        decoder_num_layers: int = 4,
        num_heads: int = 4,
        num_inputs_classes: int = 4,
        embedding_dim: int = 64,
        decoder_embedding_dim: int = 64,
        ffn_hidden_dim: int = 64,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        k: int = 1000,
        min_value: float = 0.0,
    ) -> None:

        super().__init__()

        self.embedding_dim = embedding_dim
        self.k = k
        self.min_value = min_value

        self.relu = nn.ReLU()
        self.pos_enc = PositionalEncoding(embedding_dim, dropout=encoder_dropout)
        self.linear_projection = nn.Linear(num_inputs_classes, embedding_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=encoder_dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # --- task 1: letters -------------------------------------------------
        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=decoder_dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=decoder_num_layers,
        )

        self.vocab_projection = nn.Linear(embedding_dim, vocab_size)

        # --- task 2: enc recon -----------------------------------------------
        self.reconffn1 = nn.Linear(embedding_dim, embedding_dim)
        self.reconffn2 = nn.Linear(embedding_dim, num_inputs_classes)

    def get_epsilon(
        self,
        min_value: float,
        step: int,
        k: int = 1000,
        schedule_type: str = "inverse_sigmoid",
    ) -> float:
        """
        Computes the mixing probability epsilon based on a k value and curent
        step.
        """
        if schedule_type == "linear":
            return max(0.0, 1.0 - (step / k))
        elif schedule_type == "exponential":
            return 0.999**step
        elif schedule_type == "inverse_sigmoid":
            return k / (k + math.exp(step / k))  # k / (k + exp(step / k))
        return 1.0

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        step: int,
        return_epsilon: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, float] | tuple[torch.Tensor, torch.Tensor]:

        src = self.linear_projection(src)  # (B, T_in, C) -> (B, T_in, embedding_dim)
        src = self.pos_enc(src)

        memory: torch.Tensor = self.encoder(
            src, src_key_padding_mask=src_pad_mask
        )  # (B, T_in, C)

        # --- task 1: letters -------------------------------------------------
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(
            self.embedding_dim
        )  # (B, T, C)

        tgt_emb = self.pos_enc(tgt_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_emb.size(1), device=tgt_emb.device
        ).bool()

        # --- pass 1: prediction ---
        with torch.no_grad():
            pred = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,  # causal
                memory_mask=None,  # causal
                tgt_key_padding_mask=tgt_pad_mask,  # padding
                memory_key_padding_mask=src_pad_mask,  # padding
            )  # (B, T, C)

            logits_p1 = self.vocab_projection(pred)  # (B, T, vocab_size)
            predictions = torch.argmax(logits_p1, dim=-1)  # (B, T)

        # --- mixing step ---
        epsilon = self.get_epsilon(
            min_value=self.min_value,
            step=step,
            k=self.k,
            schedule_type="linear",
        )  # get epsilon

        # fill a tensor of shape (B, T) of all epsilon values, then coin flip all
        coin_flips = torch.bernoulli(
            input=torch.full(size=tgt.shape, fill_value=epsilon, device=tgt.device)
        ).bool()  # (B, T)

        # predictions[:, i] is the model's prediction for position i+1 (causal
        # decoder output at position i has only seen tokens 0..i). Shift right
        # by one so a self-generated token lines up with the position it's
        # substituted into; position 0 has no prediction, so seed it with tgt.
        shifted_predictions = torch.empty_like(predictions)
        shifted_predictions[:, 0] = tgt[:, 0]
        shifted_predictions[:, 1:] = predictions[:, :-1]

        # make index tgt if epsilon true, otherwise shifted prediction
        mixed_tgt = torch.where(
            condition=coin_flips, input=tgt, other=shifted_predictions
        )  # (B, T)

        # make sure <SOS> exists
        mixed_tgt[:, 0] = tgt[:, 0]

        tgt_emb = self.decoder_embedding(mixed_tgt) * math.sqrt(
            self.embedding_dim
        )  # (B, T, C)

        tgt_emb = self.pos_enc(tgt_emb)

        # --- pass 2: prediction ---
        pred = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,  # causal
            memory_mask=None,  # causal
            tgt_key_padding_mask=tgt_pad_mask,  # padding
            memory_key_padding_mask=src_pad_mask,  # padding
        )  # (B, T, C)

        logits_p2: torch.Tensor = self.vocab_projection(pred)  # (B, T, vocab_size)

        # --- task 2: enc recon -----------------------------------------------
        recon = self.reconffn1(memory)  # (B, T, C)
        recon = self.relu(recon)
        recon: torch.Tensor = self.reconffn2(recon)  # (B, T, 4)

        return (logits_p2, recon, epsilon) if return_epsilon else (logits_p2, recon)
