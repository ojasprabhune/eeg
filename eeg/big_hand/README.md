# Big Hand modeling

This module focuses on modeling high-fidelity hand movements independently of EEG. It defines the "language" of hand positions through vector quantization and autoregressive sequence prediction.

## Subdirectories

### [position_llm/](position_llm/)
The core implementation of the hand position language model.
- `position_llm.py`: An autoregressive Transformer decoder that predicts sequences of discrete hand position tokens. It optionally supports duration prediction for each token.
- `e2e_position_llm.py`: An end-to-end model that combines the Position LLM with a regression head to reconstruct continuous appendage values.
- `tokenizer.py`: Implements `RegionTokenizer` (using K-Means) and `DeltaTokenizer` (using fixed binning) to convert continuous movement into discrete tokens.
- `appendage_dataset.py`: Loads and precomputes VQ-VAE tokens for training the Position LLM.
- `vqvae/`: Contains the **Vector Quantized Variational Autoencoder** used to learn the discrete codebook for hand positions.

### [cluster/](cluster/)
Utilities for unsupervised clustering of hand pose data.
- `kmeans.py`: Script to train K-Means models on appendage vectors to define the "regions" used for tokenization.
