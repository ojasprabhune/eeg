# EEG Datasets

This directory contains various PyTorch `Dataset` implementations for loading and preprocessing EEG data.

### `eeg_dataset.py`
The primary dataset used for the EEG-to-Hand task.
- Loads raw EEG (.fif) and hand (.npy) files.
- Applies standard EEG filtering (Band-pass 0.1-50Hz, Notch 60Hz, Average Reference).
- Resamples EEG to match hand data frequency.
- Pre-computes VQ-VAE tokens for hand positions.
- Implements an 80/20 train-validation split.

### `eeg_hand_cnn_dataset.py`
Specialized for CNN classification models.
- Loads motor imagery data from the PhysioNet dataset.
- Chunks EEG into epochs based on annotated events.
- Assigns discrete gesture labels (Fist, Left, Fingers, Open).

### `lstm_dataset.py`
A variant of the main EEG dataset specifically formatted for recurrent sequence modeling.

### `tokenizer.py`
Placeholder for discrete EEG tokenization (future work).
