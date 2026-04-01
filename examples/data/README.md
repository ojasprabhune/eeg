# Data loading and clustering examples

This folder contains examples demonstrating how to load hand position data, process deltas, and train clustering models for vector quantization.

### `basic_data_loading.py`
Demonstrates basic data loading from .npy files, calculating differences (deltas) between time steps, and visualizing the data using `matplotlib`.

### `test_delta_dataset.py`
Tests the `DeltaTokenizer` and `DeltaDataset` by encoding and decoding a sample sequence of deltas. It also shows how to pass tokenized data into a `PositionLLM` instance.

### `train_kmeans.py` & `train_kmeans_region.py`
These scripts use K-Means clustering to perform vector quantization on hand position data (either raw or deltas). This partitions continuous 63-channel data into discrete "regions".

#### Arguments
- `save_location`: Directory to save the resulting `.joblib` model files.
- `data_file`: Path to the `.npy` file containing the training data.
