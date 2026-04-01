# EEG data processing and modeling

This module is the core of the EEG-to-Hand movement pipeline. It provides a variety of model architectures and specialized datasets for regressing or classifying EEG signals into hand positions.

## Subdirectories

### [datasets/](datasets/)
Specialized PyTorch `Dataset` implementations:
- `eeg_dataset.py`: Main dataset for EEG-to-Hand regression, handling filtering, resampling, and VQ-VAE token pre-computation.
- `eeg_hand_cnn_dataset.py`: Dataset optimized for CNN-based classification of motor imagery.
- `eeg_hand_dataset.py`: General purpose EEG and hand label dataset.
- `lstm_dataset.py`: Dataset formatted for sequence modeling with RNNs/LSTMs.

### [eeg_llm/](eeg_llm/)
Implementation of the main **EEG Language Model**. This is a sequence-to-sequence Transformer that encodes EEG patches and decodes them into discrete hand position tokens and durations.

### [eeg_lstm/](eeg_lstm/)
Implementation of Bi-LSTM architectures for sequence modeling of EEG data, providing a temporal baseline to the Transformer-based LLM.

### [basic_models/](basic_models/)
A collection of baseline and auxiliary models:
- `eeg_cnn.py`: 2D CNN for spatial-temporal EEG feature extraction.
- `eeg_regression.py`: Basic MLP for direct value regression.
- `labram_backbone_lin.py`: Uses the LaBraM backbone with a linear classification/regression head.
