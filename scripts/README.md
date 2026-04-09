# Core training and utility scripts

This directory contains the primary training and setup scripts for the EEG-to-Hand pipeline.

## Directory Structure

### `data/`
Utilities for data collection (MediaPipe), manual labeling, synchronization (trimming), and dataset preparation.
See [data/README.md](data/README.md) for more details.

### `bionic_hand/`
Integration and control code for the Pollen Robotics AmazingHand. Includes configurations for simulation (MuJoCo), real hardware, and standalone setup scripts.
See [bionic_hand/Demo/README.md](bionic_hand/Demo/README.md) for more details.

---

## Training Scripts

### `train_vqvae.py`
Trains a Vector Quantized Variational Autoencoder (VQ-VAE) to compress 12-dimensional hand appendage vectors into discrete latent representations (tokens).

#### Usage
```bash
uv run scripts/train_vqvae.py <checkpoint_name>
```

---

### `train_temporal_model.py`
A utility script for initializing and training the `TemporalModel` on the `gesture2hand` dataset, which processes bandpower features from raw EEG for temporal classification.

#### Usage
```bash
uv run scripts/train_temporal_model.py
```
