# EEG Recurrent Models (LSTM)

This directory contains the implementation of recurrent neural network architectures for EEG sequence modeling.

### `eeg_lstm.py`
Implementation of a Bidirectional LSTM (BLSTM) model for EEG-to-Hand movement translation.
- **Preprocessing**: Uses 1D temporal convolutions to extract local features before passing to the LSTM.
- **Core**: Bidirectional LSTM layers to capture long-range temporal dependencies in the EEG signal.
- **Output**: Linear projection to the discrete hand position vocabulary.
