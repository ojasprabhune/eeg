# Data collection and preprocessing

This directory contains utilities for collecting, labeling, and preprocessing hand pose estimation data, as well as trimming multi-modal EEG and hand data to synchronized segments.

### `collect_data.py`

Collects hand pose estimation data for XYZ position coordinates for 21 joints on a hand using MediaPipe. Data is stored in an .npy file as a 2xTx63 array, where:
- Batch 0: World landmarks (metric scale).
- Batch 1: Normalized landmarks (image coordinates).
- T: Number of frames.
- 63: Flattened XYZ coordinates for 21 joints.

#### Arguments
- `filename`: The base name for the output .npy file.
- `data_time`: Length of collection in seconds.
- `--webcam`: Boolean, if True, uses webcam (default: False).
- `--input_video`: Path to a video file if not using webcam.
- `--save_video`: Boolean, if True, saves the processed video to `data/videos/`.
- `--plot`: Boolean, if True, plots joint data after collection.
- `--joint`: Specific joint to plot (e.g., "W" for wrist, "IT" for index tip).

### `annotate_video.py`

An interactive utility to manually label video frames with gesture classes.

#### Usage
- Press keys **1-4** to assign gesture labels:
    1. Fist
    2. Left
    3. Fingers
    4. Open
- Controls:
    - `s`: Toggle 2x playback speed.
    - `p`: Pause/Resume.
    - `d`: Jump back 5 seconds.
    - `q`: Save and quit.

#### Arguments
- `video_path`: Path to the input video file.
- `output_npy`: Path to save the resulting labels as an .npy file.

### `trim_video.py`

Trims and synchronizes EEG data (.edf), hand position data (.npy), and labels (.npy) to a common start point and duration.

#### Arguments
- `eeg_data_path`: Base path to EEG file (exclude .edf).
- `hand_data_path`: Base path to hand data file (exclude .npy).
- `labels_path`: Base path to labels file (exclude .npy).
- `eeg_trim_start_seconds`: Start time to trim from in the EEG file.
- `hand_trim_start_seconds`: Start time to trim from in the hand data file.
- `start_offset_seconds`: Additional offset applied to both.

### `save_data.py`

Aggregates all .npy files in the `data/` directory, performs an 80/20 train-validation split along the time dimension, and saves the resulting datasets to `data/dataset/training_dataset.npy` and `data/dataset/validation_dataset.npy`.

### `train_kmeans_appendage.py`

Trains a K-Means clustering model on hand appendage vectors (derived from joint positions) for vector quantization into discrete "region" tokens.

#### Arguments
- `save_location`: Directory where `kmeans.joblib` and `kmeans_scaler.joblib` will be saved.
