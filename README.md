# NORA: A neural decoding model to regress EEG to hand movements

A neural output regression algorithm consisting of a new sequence-to-sequence Transformer to regress electroencephalogram data to hand movements. To reduce EEG noise, we trained a next hand position prediction decoder model to bias the EEG model’s probability distributions. EEG data collected using Emotiv Epoc X EEG headset.

## Getting Started

### 1. Installation
First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/ojasprabhune/eeg.git
cd eeg
```

### 2. Download Models
Download the required EEG and position prediction models from [this placeholder link](models/) and place them in the `models/` directory.

### 3. Setup Environment
We use `uv` for fast dependency management. Sync the environment and install the package in editable mode:
```bash
uv sync
uv pip install -e .
```

## Running the Bionic Hand

Navigate to the bionic hand demo directory:
```bash
cd scripts/bionic_hand/Demo
```

You can control the hand using two primary methods:

### Option A: Webcam Control (Real-time)
To control the bionic hand in real-time using your webcam and MediaPipe hand tracking:
```bash
dora up
dora build dataflow_tracking_real.yml --uv
dora run dataflow_tracking_real.yml --uv
```
This runs `scripts/bionic_hand/Demo/HandTracking/HandTracking/main.py` as part of the dataflow.

### Option B: Preprocessed JointData Control
You can also control the hand using a pre-recorded `.npy` file containing joint positions.
1. Collect data using the data collection script:
   ```bash
   python scripts/data/collect_data.py my_movement 10 --webcam True
   ```
2. Run the dataflow with the preprocessed file:
   ```bash
   dora up
   dora build dataflow_real.yml --uv
   dora run dataflow_real.yml --uv
   ```
   *Note: You may need to edit `dataflow_real.yml` to specify the correct path to your `.npy` file if not using the default.*

### Platform Requirements & Hardware
- **OS:** This setup is pre-configured for **macOS**.
- **Serial Port:** The AmazingHand serial board is mapped to `/dev/ttyAMC0` in the YAML dataflow files.
- **Windows Users:** If you are running on Windows, the board will likely appear as a COM port (e.g., `COM13`). You must edit `dataflow_real.yml` and `dataflow_tracking_real.yml` to reflect your specific port.
