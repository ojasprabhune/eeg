# EEG Gesture Classification

This module contains datasets and models for classifying EEG signals into gesture classes or movement states. It supports raw EEG, bandpower, CSP, and DWT features. The output is a class label, not a hand position or appendage vector.

## Subdirectories

### [datasets/](datasets/)
Handles EEG data loading, preprocessing, and feature extraction.
- `self/`: Datasets for self-recorded Emotiv EPOC X EEG.
  - `gesture_dataset.py`: `GestureDataset` loads XDF recordings and extracts labeled movement epochs from cue markers, with raw EEG, bandpower, CSP, and DWT features.
  - `temporal_dataset.py`: `TemporalDataset` loads continuous FIF recordings and saved class labels, aligns EEG and bandpower features, and extracts sliding windows with configurable stride. It returns EEG, bandpower features, and one majority class label per window.
- `existing/`: Datasets from public sources.
  - `physio_net_gesture_dataset.py`: `PhysioNetGestureDataset` loads the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB). It uses 160 Hz executed-movement recordings for four classes: left fist, right fist, both fists, and both feet. It provides raw EEG, bandpower, CSP, and DWT features and supports loading saved epochs.
- `utils.py`: Shared bandpower and DWT feature extraction, Emotiv channel names, and terminal colors.

### [models/](models/)
Contains custom models and implementations of published architectures.
- `self/`: Custom classification models.
  - `gesture_model.py`: `GestureModel` uses a Transformer encoder and a decoder with one learned query to classify an entire epoch.
  - `gesture_temporal_model.py`: `GestureTemporalModel` uses a Transformer encoder and attention-weighted pooling to produce one gesture prediction per epoch.
  - `temporal_model.py`: `TemporalModel` processes sequences of bandpower features with a Transformer encoder and attention-weighted pooling for movement-state classification.
  - `linear_baseline.py`: `EEGLinearBaseline` mean-pools features over time and applies a small feedforward classification head.
  - `transformer/`: Shared positional encoding used by the Transformer models.
- `existing/`: Implementations of published architectures.
  - `eegnet.py`: `EEGNet` is a PyTorch implementation adapted from EEGNet. It uses temporal, depthwise spatial, and separable convolutions to classify raw EEG, with input shape `(batch_size, 1, num_channels, num_samples)`.

### [utils/](utils/)
Defines gesture experiment mappings and helpers for converting letters to gesture classes.
