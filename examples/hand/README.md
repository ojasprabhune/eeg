# Hand movement and classification examples

This folder contains examples for training and testing models related to hand pose estimation, classification, and movement prediction using Transformers (LLMs).

### `big_hand_llm.py`
Trains a `PositionLLM` to generalize Ojas's hand movements. It utilizes VQVAE tokenization on appendage vector components and is configured via `config/log_position_llm.yaml`.

#### Requirements
- Ensure `config/log_position_llm.yaml` exists and contains correct hyperparameters and paths.
- Requires a GPU (`cuda`) for training.

### `test_hand_classifying.py`
Runs a live webcam demo that uses MediaPipe to extract 21 hand landmarks (63 joints) and passes them through a pre-trained `HandClassifier` to predict the hand's current class (1-4).

#### Running the script
```bash
uv run examples/hand/test_hand_classifying.py
```
- **Checkpoint:** By default, it looks for the model checkpoint at `models/hand_classifier/hand_classifier_epoch_100.pth`.
- **Exit:** Press 'q' to close the webcam window.

### `train_e2e.py`
Demonstrates an end-to-end (E2E) training loop for a `PositionLLM` using `RegionTokenizer` and `DeltaTokenizer`. It trains on a specific recording (`data/open_fist_front.npy`) to predict the next region and delta tokens.

#### Paths
- Ensure `models/delta_tokens` exists for the `RegionTokenizer`.
- Ensure `data/open_fist_front.npy` is available in the data directory.
