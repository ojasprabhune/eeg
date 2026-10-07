# Gesture to Hand Models

This directory contains the model implementations for identifying hand gestures and movement states from temporal EEG features.

### `gesture_model.py`

A Transformer encoder-decoder model that classifies an entire EEG epoch into one gesture class.

- **Input Projection**: Maps EEG feature vectors to the model dimension and adds positional encoding.
- **Transformer Encoder**: Processes the feature sequence over time.
- **Learned Query Decoder**: Uses one learned query to attend to the encoded epoch and produce a pooled summary.
- **Classification Head**: Maps that summary to gesture-class logits.

### `gesture_temporal_model.py`

An encoder-only alternative to `GestureModel` that produces one gesture prediction per epoch.

- **Input Projection**: Maps EEG feature vectors to the model dimension and adds positional encoding.
- **Transformer Encoder**: Captures the temporal dynamics of the feature sequence.
- **Attention Pooling**: Weights the encoded time steps and combines them into one summary.
- **Classification Head**: Maps that summary to gesture-class logits without a decoder.

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
