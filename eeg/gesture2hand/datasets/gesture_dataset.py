import re
from pathlib import Path

import mne
import numpy as np
import pyxdf
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ..utils import gesture_experiments
from .utils import EMOTIV_CHANNELS, Colors, compute_bandpower_features


class GestureDataset(Dataset):
    """
    Loads raw LSL/EmotivPRO recordings (.xdf) and slices out one epoch per
    motor-execution trial, each epoch labeled with the gesture class that was
    cued. Returns both a raw filtered-channel epoch and a bandpower epoch for
    every trial, so models can be trained on either input independently.

    See notebooks/lsl.ipynb for the event/epoch extraction this is based on,
    and scripts/data/eeg_scheduling/cueing.py for how markers get written
    during recording: REST_START, CUE_{class}, PRE_{class}, MOVE_{class},
    END_SEQ, SESSION_END, where {class} is the 1-indexed gesture class.
    """

    def __init__(
        self,
        experiment: str,
        recordings_path: str = "/Users/ojasprabhune/Documents/research/NORA/recordings/lsl",
        mode: str = "train",
        val_ratio: float = 0.2,
        motor_exec_sec: float = 3.0,
        bp_window_sec: float = 1.0,
        bp_step_samples: int = 4,
        verbose: bool = False,
    ) -> None:
        """
        Input is every recorded .xdf session under recordings_path. Each
        MOVE_{class} marker becomes one epoch spanning motor_exec_sec seconds
        (matching cueing.py's MOTOR_EXEC), labeled with class - 1.
        """

        print(
            f"{Colors.HEADER}{Colors.BOLD}Initializing gesture dataset...{Colors.ENDC}"
        )
        self.verbose = verbose
        self.mode = mode
        self.motor_exec_sec = motor_exec_sec

        super().__init__()

        self.num_classes = max(gesture_experiments[experiment].values())

        # --- load + epoch every session ---------------------------------
        print(f"{Colors.OKBLUE}Getting recordings...{Colors.ENDC}")
        session_paths = sorted(Path(recordings_path).rglob("*_eeg.xdf"))

        all_raw_epochs, all_bp_epochs, all_labels = [], [], []

        for path in session_paths:
            raw_epochs, bp_epochs, labels = self._epoch_session(
                path, bp_window_sec=bp_window_sec, bp_step_samples=bp_step_samples
            )
            if len(labels) == 0:
                continue
            all_raw_epochs.append(raw_epochs)
            all_bp_epochs.append(bp_epochs)
            all_labels.append(labels)

        if not all_labels:
            raise RuntimeError(f"No MOVE_* epochs found under {recordings_path}")

        # different sessions can round to slightly different epoch lengths,
        # so trim every session to the shortest T before stacking
        min_raw_t = min(e.shape[1] for e in all_raw_epochs)
        min_bp_t = min(e.shape[1] for e in all_bp_epochs)

        self.raw_epochs = np.concatenate(
            [e[:, :min_raw_t, :] for e in all_raw_epochs], axis=0
        ).astype(np.float32)  # (N, T_raw, 14)
        self.bp_epochs = np.concatenate(
            [e[:, :min_bp_t, :] for e in all_bp_epochs], axis=0
        ).astype(np.float32)  # (N, T_bp, 84)
        self.labels = np.concatenate(all_labels, axis=0).astype(np.int64)  # (N,)

        print(
            f"{Colors.OKGREEN}Loaded {len(self.labels)} epochs from "
            f"{len(session_paths)} session(s).{Colors.ENDC}"
        )

        if verbose:
            print(Colors.HEADER)
            print("Raw epochs shape:       ", self.raw_epochs.shape)
            print("Bandpower epochs shape: ", self.bp_epochs.shape)
            print("Labels shape:           ", self.labels.shape)
            print(Colors.ENDC)

        # --- train/val split (stratified by class) -----------------------
        # each epoch is an isolated trial with rest periods on either side,
        # so unlike TemporalDataset's sliding windows there's no adjacency
        # leakage to worry about — a plain per-class random split is enough
        rng = np.random.RandomState(42)
        train_idx, val_idx = [], []
        for cls in range(self.num_classes):
            cls_idx = np.where(self.labels == cls)[0]
            rng.shuffle(cls_idx)
            n_val = int(len(cls_idx) * val_ratio) if len(cls_idx) > 1 else 0
            val_idx.extend(cls_idx[:n_val])
            train_idx.extend(cls_idx[n_val:])

        self.train_idx = np.array(sorted(train_idx), dtype=np.int64)
        self.val_idx = np.array(sorted(val_idx), dtype=np.int64)

        print(
            f"{Colors.OKBLUE}Split: {len(self.train_idx)} train, "
            f"{len(self.val_idx)} val{Colors.ENDC}\n"
        )

    def _epoch_session(
        self, path: Path, bp_window_sec: float, bp_step_samples: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads one .xdf recording and slices it into per-trial epochs.

        Returns
        -------
        raw_epochs : np.ndarray, shape (N, T_raw, 14)
        bp_epochs  : np.ndarray, shape (N, T_bp, 84)
        labels     : np.ndarray, shape (N,)
            Zero-based gesture class per epoch.
        """
        streams, _ = pyxdf.load_xdf(str(path))

        # find streams by type rather than assuming a fixed order — pyxdf
        # doesn't guarantee the EEG stream comes before the marker stream
        eeg_stream = next(s for s in streams if s["info"]["type"][0] == "EEG")
        marker_stream = next(s for s in streams if s["info"]["type"][0] == "Markers")

        sfreq = float(eeg_stream["info"]["nominal_srate"][0])
        labels_in_file = [
            ch["label"][0]
            for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        ]
        channel_idx = [labels_in_file.index(ch) for ch in EMOTIV_CHANNELS]

        data = eeg_stream["time_series"].T.astype(np.float64)[channel_idx]  # (14, T)

        info = mne.create_info(ch_names=EMOTIV_CHANNELS, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)

        # same filtering pipeline as TemporalDataset: wideband, line-noise
        # notch, common average reference
        raw.filter(l_freq=0.1, h_freq=50, verbose=False)
        raw.notch_filter(freqs=60, verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)

        filtered: NDArray = raw.get_data().T  # (T, 14)

        bp_features = compute_bandpower_features(
            filtered,
            sfreq=sfreq,
            window_sec=bp_window_sec,
            step_samples_128=bp_step_samples,
        )  # (T_bp, 84)
        bp_rate = sfreq / bp_step_samples
        nperseg = int(bp_window_sec * sfreq)
        bp_offset = (nperseg // 2) / sfreq
        bp_times = bp_offset + np.arange(len(bp_features)) / bp_rate

        # --- slice one epoch per MOVE_{class} marker ----------------------
        eeg_start = eeg_stream["time_stamps"][0]
        t_raw = round(self.motor_exec_sec * sfreq)
        t_bp = round(self.motor_exec_sec * bp_rate)

        raw_epochs, bp_epochs, labels = [], [], []

        for t, marker in zip(
            marker_stream["time_stamps"], marker_stream["time_series"]
        ):
            match = re.fullmatch(r"MOVE_(\d+)", marker[0])
            if match is None:
                continue

            cls = int(match.group(1)) - 1  # zero-based, matches get_gesture_class

            onset_time = t - eeg_start
            onset_raw = round(onset_time * sfreq)
            onset_bp = int(np.searchsorted(bp_times, onset_time))

            if onset_raw + t_raw > len(filtered) or onset_bp + t_bp > len(bp_features):
                continue  # trial got cut off at the end of the recording

            raw_epochs.append(filtered[onset_raw : onset_raw + t_raw])
            bp_epochs.append(bp_features[onset_bp : onset_bp + t_bp])
            labels.append(cls)

        if not labels:
            return (
                np.zeros((0, t_raw, 14), dtype=np.float32),
                np.zeros((0, t_bp, 84), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        return np.stack(raw_epochs), np.stack(bp_epochs), np.array(labels)

    def __len__(self) -> int:
        return len(self.train_idx) if self.mode == "train" else len(self.val_idx)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the raw-channel epoch, bandpower epoch, and gesture-class
        label at the given index from the training or validation split.
        """
        split_idx = self.train_idx if self.mode == "train" else self.val_idx
        i = split_idx[index]

        return (
            torch.tensor(self.raw_epochs[i]),
            torch.tensor(self.bp_epochs[i]),
            torch.tensor(self.labels[i]),
        )

    def get_sampler_weights(self) -> tuple[list[float], torch.Tensor]:
        """
        Per-sample and per-class weights for a WeightedRandomSampler /
        CrossEntropyLoss, computed over the current split.
        """
        split_idx = self.train_idx if self.mode == "train" else self.val_idx
        split_labels = self.labels[split_idx]

        class_counts = np.bincount(split_labels, minlength=self.num_classes)
        total_samples = len(split_labels)
        weights_per_class = total_samples / (self.num_classes * (class_counts + 1e-8))

        sample_weights = [
            float(weights_per_class[int(label)]) for label in split_labels
        ]
        class_weights_tensor = torch.tensor(weights_per_class, dtype=torch.float32)

        return sample_weights, class_weights_tensor
