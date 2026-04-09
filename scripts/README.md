# Core training and utility scripts

This directory contains the primary training scripts for the EEG-to-Hand pipeline, including VQ-VAE for hand pose quantization and various LLM/Transformer architectures for sequence modeling.

## Directory Structure

### `data/`
Utilities for data collection (MediaPipe), manual labeling, synchronization (trimming), and dataset preparation.
See [data/README.md](data/README.md) for more details.

### `bionic_hand/`
Integration and control code for the Pollen Robotics AmazingHand. Includes configurations for simulation (MuJoCo) and real hardware.
See [bionic_hand/Demo/README.md](bionic_hand/Demo/README.md) for more details.

---

## Training Scripts

### `train_vqvae.py`
Trains a Vector Quantized Variational Autoencoder (VQ-VAE) to compress 12-dimensional hand appendage vectors into discrete latent representations (tokens). This allows the downstream LLM to operate on a discrete vocabulary rather than continuous values.

#### Usage
```bash
uv run scripts/train_vqvae.py <checkpoint_name>
```

---

### `train_big_eeg_llm.py`
A main training script for the sequence-to-sequence model that regresses EEG signals directly to hand appendage tokens and durations. It incorporates pre-trained LaBraM backbones and the VQ-VAE tokenizer.

---

### `train_temporal_model.py`
A utility script for initializing and testing the `TemporalDataset` from the `gesture2hand` module, which prepares bandpower features from raw EEG for temporal classification.

---

### `train_e2e.py` (Legacy/Reference)
An end-to-end training script for the autoregressive Position LLM, used for validating the discrete tokenization and next-position prediction pipeline.
