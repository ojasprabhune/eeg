# Scripts

This directory contains training scripts and utilities for the EEG and hand pose estimation pipeline.

## Directory Structure

### `data/`

Data collection and preprocessing utilities for hand pose estimation.

#### `collect_data.py`

Collects hand pose estimation data for XYZ position coordinates for 21 joints on a hand. Data is stored in an .npy file with T (number of frames) rows and 63 columns (channels).

#### `save_data.py`

Saves processed hand pose estimation data.

#### `trim_video.py`

Utility for trimming video files.

#### `train_kmeans_appendage.py`

Trains K-means clustering model on appendage (finger/hand) data for tokenization.

---

### `train_big_eeg_llm.py`

Trains a large EEG language model end-to-end. This is a main training script for the EEG LLM pipeline.

The EEG LLM is a sequence-to-sequence model, regressing EEG data to appendage values for applying to kinematic hand movements.

---

### `train_e2e.py`

Trains a large hand position language model end-to-end. This is a main training script for the Position LLM pipeline.

It consists of an autoregressive transformer decoder and uses appendage values and VQ-VAE tokens.

---

### `train_vqvae.py`

Trains a Vector Quantized Variational Autoencoder (VQ-VAE) for learning discrete latent representations of 12 appendage values.

---

### `bionic_hand/`

Integration and control code for the bionic hand hardware and simulation.

#### `Demo/`

Demonstration and simulation configurations for the bionic hand.

**Key Files:**
- `dataflow_*.yml` - Dataflow configuration files for different modes (simulation, real hardware, tracking)
- `README.md` - Bionic hand demo documentation
