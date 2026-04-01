# Gesture to Hand mapping

This module contains datasets and models for temporal classification of EEG signals into hand gestures or movement states. It focuses on extracting frequency-domain features (bandpower) and modeling their dynamics over time.

## Subdirectories

### [datasets/](datasets/)
Handles EEG data loading, preprocessing, and feature extraction.
- `temporal_dataset.py`: Extracts bandpower features (Theta, Mu, Beta, Gamma) from raw EEG and aligns them with hand position data. It supports sliding window extraction with configurable stride.
- `utils.py`: Contains common utilities for data normalization and appendage vector calculations.

### [models/](models/)
Contains temporal modeling architectures.
- `temporal_model.py`: A Transformer-based architecture that processes sequences of bandpower features. It uses attention-weighted pooling to identify critical time steps for gesture classification.
- `transformer/`: Lower-level Transformer building blocks (Attention, Encoder, Decoder).
