# Examples

This directory contains experimental scripts and usage examples for the various components of the EEG and hand modeling pipeline.

## Subdirectories

### [data/](data/README.md)
Examples for data loading, preprocessing, and training K-Means models for quantization.

### [eeg/](eeg/)
Scripts for training models on EEG data, including:
- `train_big_eeg_llm.py`: Main sequence-to-sequence model.
- `train_eeg_cnn.py`: Convolutional architecture for EEG classification.
- `train_eeg_lstm.py`: Bi-LSTM architecture for EEG processing.
- `train_basic_labram.py`: Linear models using LaBraM backbones.
- `convert_labram.py`: Utility to map weights from original LaBraM checkpoints.

### [hand/](hand/)
Scripts for training models on hand position data:
- `big_hand_llm.py`: Autoregressive Position LLM with duration prediction.
- `train_e2e.py`: End-to-end training reference script.

### [overfit/](overfit/README.md)
Small-scale training scripts designed to overfit models for validation purposes.

### [region_tokens/](region_tokens/README.md)
Examples focused on the region tokenization process and related tokenizer tests.
