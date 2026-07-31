# EEG Language Model (EEG-LLM)

This directory contains the implementation of the main sequence-to-sequence Transformer model for EEG-to-Hand movement translation.

### `eeg_llm.py`
The main architecture:
- **Input**: EEG data of shape (B, T, num_channels).
- **Encoder**: Processes EEG patches to extract latent temporal features.
- **Decoder**: An autoregressive Transformer decoder that uses the encoded EEG features (cross-attention) to predict the next hand position token.
- **Duration Head**: An auxiliary MLP that predicts the duration of the current hand position state.

### `transformer/`
Low-level Transformer implementations optimized for EEG modeling.
