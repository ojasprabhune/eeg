# Gesture to Hand Models

This directory contains the model implementations for identifying hand gestures and movement states from temporal EEG features.

### `temporal_model.py`

A sequence-to-sequence classification model designed for bandpower features.

- **Input Projection**: Maps high-dimensional bandpower vectors to a latent model dimension.
- **Transformer Encoder**: Captures the temporal dynamics of the frequency features (e.g., Mu-desynchronization).
- **Attention Pooling**: Learns to weight the importance of different time steps in the sequence for the final gesture classification.

### `linear_baseline.py`

A simple linear classifier that takes mean-pooled bandpower features as input. This serves as more of a baseline model and benchmark for comparison against the larger model above.

It achieves a 71% accuracy predicting only 2 classes (rest vs movement).

### `transformer/`

Shared Transformer layers used for temporal sequence modeling.
