# VQ-VAE for Hand Positions

This directory contains the implementation of the Vector Quantized Variational Autoencoder (VQ-VAE) used to quantize continuous 12-dimensional hand appendage vectors into discrete tokens.

### `vqvae.py`
The top-level model that orchestrates the encoding, quantization (using the codebook), and decoding processes.

### `vqvae_encoder.py`
A Transformer-based encoder that processes sequences of hand positions and outputs continuous latent embeddings. It includes the logic for finding the nearest codebook entry (quantization) and updating the codebook using Exponential Moving Averages (EMA).

### `vqvae_decoder.py`
A Transformer-based decoder that takes quantized embeddings and reconstructs the original continuous 12-dimensional appendage vectors.

### `transformer/`
Contains the shared Transformer layers (Attention, Encoder blocks) used by both the VQ-VAE encoder and decoder.
