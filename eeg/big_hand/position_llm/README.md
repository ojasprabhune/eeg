# Position LLM components

This directory contains the core logic for the Hand Position Language Model (Position LLM).

### `position_llm.py`
The main autoregressive model.
- Uses a Transformer Decoder (without cross-attention).
- Predicts the next discrete hand position token from a sequence.
- Supports an optional MLP head for **Duration Prediction**, allowing the model to estimate how long a hand remains in a particular quantized state.

### `e2e_position_llm.py`
Wraps the Position LLM with an additional linear layer to map predicted discrete logits back to 12-dimensional appendage vectors, enabling end-to-end continuous reconstruction.

### `tokenizer.py`
Implements quantization logic:
- `RegionTokenizer`: Maps 12D vectors to 50 clusters defined by K-Means.
- `DeltaTokenizer`: Maps 63D delta values to a fixed vocabulary of discrete steps (e.g., -10.0 to 10.0 in 0.1 increments).

### `appendage_dataset.py`
Loads synchronized hand position data and pre-computes VQ-VAE tokens for efficient training. It handles the windowing of long sequences into manageable chunks for the Transformer.

### `vqvae/`
Contains the **VQ-VAE (Vector Quantized Variational Autoencoder)** implementation, which is used to learn the discrete latent codebook from continuous hand trajectories.
