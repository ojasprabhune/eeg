# Overfitting examples

This directory contains scripts focused on overfitting various model architectures to small datasets. This is used to validate model capacity and the correctness of the training pipeline (e.g., loss calculation, gradient flow).

### `train_overfit.py`
Trains a basic `PositionLLM` to overfit on a single joint's delta tokens.

### `train_overfit_regions_deltas.py`
Trains a `PositionLLM` to overfit on 63-channel delta tokens that have been quantized into region tokens using K-Means.

### `train_e2e_overfit.py`
An end-to-end overfitting script that predicts next region tokens and simultaneously optimizes for the underlying delta tokens using a combined loss function.

#### Overview:
1. **Raw positions** (T, 63) &rarr; Normalized and tokenized to deltas.
2. **Delta tokens** (T, 63) &rarr; Quantized into region tokens.
3. **Region tokens** (T,) &rarr; Fed into Position LLM.
4. **Output**: Next-region probabilities and reconstructed delta probabilities.

#### Combined Loss:
$L = \lambda L_{region}(\hat r_{tokens},\ r_{gt}) + \lambda L_{delta}(\hat d_{tokens},\ d_{gt})$
