# Data collection and preprocessing

This directory contains utilities for collecting, labeling, and preprocessing hand pose estimation data, as well as trimming multi-modal EEG and hand data to synchronized segments.

## Directory Structure

### `hands/`
Scripts specifically for hand pose data collection, annotation, and classification.

- **`collect_data.py`**: Collects hand pose estimation data for XYZ position coordinates for 21 joints on a hand using MediaPipe. Data is stored in an .npy file as a 2xTx63 array (Batch 0: World landmarks, Batch 1: Normalized landmarks).
- **`annotate_video.py`**: An interactive utility to manually label video frames with gesture classes (1-4).
- **`train_hand_classifier.py`**: Trains a simple neural network to classify hand poses based on 63-channel joint data.
- **`run_hand_classifying.py`**: A live webcam demo that uses MediaPipe and a trained `HandClassifier` model to predict gestures in real-time.

### `util/`
General utility scripts for data management and synchronization.

- **`trim_video.py`**: Trims and synchronizes EEG data (.edf), hand position data (.npy), and labels (.npy) to a common start point and duration.
- **`save_data.py`**: Aggregates all .npy files in the `data/` directory, performs an 80/20 train-validation split, and saves the resulting datasets.

---

## Detailed Usage

### `hands/collect_data.py`
#### Arguments
- `filename`: The base name for the output .npy file.
- `data_time`: Length of collection in seconds.
- `--webcam`: Boolean, if True, uses webcam (default: False).
- `--input_video`: Path to a video file if not using webcam.
- `--save_video`: Boolean, if True, saves the processed video to `data/videos/`.
- `--plot`: Boolean, if True, plots joint data after collection.
- `--joint`: Specific joint to plot (e.g., "W" for wrist, "IT" for index tip).

### `hands/annotate_video.py`
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

### `util/trim_video.py`
#### Arguments
- `eeg_data_path`: Base path to EEG file (exclude .edf).
- `hand_data_path`: Base path to hand data file (exclude .npy).
- `labels_path`: Base path to labels file (exclude .npy).
- `eeg_trim_start_seconds`: Start time to trim from in the EEG file.
- `hand_trim_start_seconds`: Start time to trim from in the hand data file.
- `start_offset_seconds`: Additional offset applied to both.

### `train_kmeans_appendage.py`
Trains a K-Means clustering model on hand appendage vectors (derived from joint positions) for vector quantization into discrete "region" tokens.

#### Arguments
- `save_location`: Directory where `kmeans.joblib` and `kmeans_scaler.joblib` will be saved.
