import time
import warnings
from pathlib import Path

import mne
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ..utils import (
    Colors,
    compute_bandpower_features,
    compute_dwt_features,
)

np.seterr(all="ignore")


class PhysioNetGestureDataset(Dataset):
    """
    This dataset loads the PhysioNet EEG Motor Movement/Imagery Dataset and its
    EEGMMIDB EDF recordings and extracts executed-movement epochs from the
    T1/T2 annotations. T1/T2 labels depend on the run: unilateral runs are
    left/right fist, and bilateral runs are both fists/both feet.

    The gestures thus only apply to the 4 class experiments, where the classes
    are left fist, right fist, both fists, and both feet.

    The dataset returns raw, bandpower, CSP, and DWT features with a
    zero-based class label, just like GestureDataset.
    """

    def __init__(
        self,
        recordings_path: str = "/Users/ojasprabhune/Documents/research/NORA/recordings/physio_net",
        save_path: str = "/Users/ojasprabhune/Documents/research/NORA/recordings/physio_net/dataset",
        mode: str = "train",
        split: str = "subject",
        num_recordings: int = -1,
        load_from_saved: bool = True,
        motor_exec_sec: float = 2.5,
        epoch_start_sec: float = 0.5,
        bp_window_sec: float = 1.0,
        bp_step_samples: int = 4,
        verbose: bool = False,
    ) -> None:
        """
        Reads executed-movement EDF runs under recordings_path. The four
        labels are left fist, right fist, both fists, and both feet.

        The pipeline is like this:
            1. ICA cleans the recording before any epochs are taken.
            2. Bandpower finds features over time, which then get sliced.
            3. DWT takes each sliced movement epoch, and finds the features.
            4. CSP fits on the trained epochs.

        If load_from_saved is True, the dataset will load from the saved path
        instead of reprocessing the EDF files. If load_from_saved is False, the
        dataset will reprocess the EDF files and save them to the save_path.

        If num_recordings is -1, all recordings are used. If num_recordings is
        specified, then only that many recordings are used after getting EDF.
        """

        print(
            f"{Colors.HEADER}{Colors.BOLD}Initializing gesture dataset...{Colors.ENDC}"
        )
        self.verbose = verbose
        self.mode = mode
        self.motor_exec_sec = motor_exec_sec
        self.epoch_start_sec = epoch_start_sec
        self.bp_window_sec = bp_window_sec
        self.bp_step_samples = bp_step_samples

        super().__init__()

        self.num_classes = 4  # left fist, right fist, both fists, both feet

        # --- load + epoch executed runs --------------------------------------

        # labels: left fist, right fist, both fists, both feet
        run_labels = {
            "03": {"T1": 0, "T2": 1},
            "07": {"T1": 0, "T2": 1},
            "11": {"T1": 0, "T2": 1},
            "05": {"T1": 2, "T2": 3},
            "09": {"T1": 2, "T2": 3},
            "13": {"T1": 2, "T2": 3},
        }

        # path.stem is the filename without the .edf extension, so
        # path.stem[-2:] is the run number
        recording_paths = sorted(Path(recordings_path).rglob("*.edf"))
        recording_paths = [
            path for path in recording_paths if path.stem[-2:] in run_labels
        ]

        if num_recordings != -1:
            recording_paths = recording_paths[:num_recordings]

        if load_from_saved:  # load from saved path instead of reprocessing EDF files
            self.raw_epochs = np.load(Path(save_path) / "raw_epochs.npy")
            self.bp_epochs = np.load(Path(save_path) / "bp_epochs.npy")
            self.dwt_epochs = np.load(Path(save_path) / "dwt_epochs.npy")
            self.csp_epochs = np.load(Path(save_path) / "csp_epochs.npy")
            self.labels = np.load(Path(save_path) / "labels.npy")
            self.subject_ids = np.load(Path(save_path) / "subject_ids.npy")

            print(
                f"{Colors.OKGREEN}Loaded {len(self.labels)} epochs from "
                f"{save_path}{Colors.ENDC}"
            )

        else:
            start = time.time()

            print(
                f"{Colors.OKBLUE}Getting {len(recording_paths)} recordings...{Colors.ENDC}"
            )

            raw_epochs, bp_epochs, dwt_epochs, labels, subject_ids = [], [], [], [], []

            for path in recording_paths:
                run = path.stem[-2:]
                recording_epochs = self.epoch_recording(path, run_labels[run])
                raw_epochs.extend(recording_epochs[0])
                bp_epochs.extend(recording_epochs[1])
                dwt_epochs.extend(recording_epochs[2])
                labels.extend(recording_epochs[3])

                # the subject id is the first 4 characters of the filename
                n_new = len(recording_epochs[3])
                subject_ids.extend([path.stem[:4]] * n_new)

            elapsed = time.time() - start

            print(
                f"{Colors.OKGREEN}Took {elapsed:.0f} seconds to get "
                f"{len(recording_paths)} recordings...{Colors.ENDC}"
            )

            self.raw_epochs = np.stack(raw_epochs).astype(np.float32)
            self.bp_epochs = np.stack(bp_epochs).astype(np.float32)
            self.dwt_epochs = np.log(
                np.stack(dwt_epochs).astype(np.float64) + 1e-6
            ).astype(np.float32)
            self.labels = np.array(labels, dtype=np.int64)
            self.subject_ids = np.array(subject_ids)

            # range should be hundreds of uV, not 1e-5
            for name, arr in [
                ("raw", self.raw_epochs),
                ("bp", self.bp_epochs),
                ("dwt", self.dwt_epochs),
            ]:
                n_bad = (~np.isfinite(arr)).sum()
                assert n_bad == 0, f"{name} epochs have {n_bad} non-finite values!"

            if verbose:
                print(
                    "Raw epoch uV range:", self.raw_epochs.min(), self.raw_epochs.max()
                )

            # raw: (N, T_raw, 14)
            # bp: (N, T_bp, 84)
            # dwt: (N, 84)
            # labels: (N,)
            # subject_ids: (N,)

            print(f"{Colors.OKGREEN}Loaded {len(self.labels)} epochs.{Colors.ENDC}")

        # --- dataset splitting -----------------------------------------------

        if split == "stratified":
            rng = np.random.RandomState(42)
            train_idx, val_idx = [], []

            # 80% train, 20% val, same number of each class in each split
            for cls in range(self.num_classes):
                cls_idx = np.where(self.labels == cls)[0]
                rng.shuffle(cls_idx)

                split_idx = round(len(cls_idx) * 0.8)

                train_idx.extend(cls_idx[:split_idx])
                val_idx.extend(cls_idx[split_idx:])

            self.train_idx = np.array(sorted(train_idx), dtype=np.int64)
            self.val_idx = np.array(sorted(val_idx), dtype=np.int64)

        elif split == "subject":
            # 80% train, 20% val, same number of each subject in each split
            unique_subjects = np.unique(self.subject_ids)
            rng = np.random.RandomState(42)
            rng.shuffle(unique_subjects)

            split_idx = round(len(unique_subjects) * 0.8)
            train_subjects = unique_subjects[:split_idx]
            val_subjects = unique_subjects[split_idx:]

            self.train_idx = np.where(np.isin(self.subject_ids, train_subjects))[0]
            self.val_idx = np.where(np.isin(self.subject_ids, val_subjects))[0]

        print(
            f"{Colors.OKBLUE}{len(self.train_idx)} train, "
            f"{len(self.val_idx)} val{Colors.ENDC}"
        )

        # --- CSP spatial-filter features -------------------------------------

        print(f"\n{Colors.OKBLUE}Fitting CSP...{Colors.ENDC}")

        csp_input = self.raw_epochs.transpose(0, 2, 1).astype(np.float64)

        csp = mne.decoding.CSP(n_components=6, reg="ledoit_wolf", log=True)

        with mne.utils.use_log_level("warning"):
            csp.fit(csp_input[self.train_idx], self.labels[self.train_idx])
            self.csp_epochs = csp.transform(csp_input).astype(np.float32)  # (N, 6)

        # range should be hundreds of uV, not 1e-5
        n_bad = (~np.isfinite(self.csp_epochs)).sum()
        assert n_bad == 0, f"CSP epochs have {n_bad} non-finite values!"
        print("CSP epoch uV range:", self.csp_epochs.min(), self.csp_epochs.max())

        if verbose:
            print(Colors.HEADER)
            print("Raw epochs shape:        ", self.raw_epochs.shape)
            print("Bandpower epochs shape:  ", self.bp_epochs.shape)
            print("DWT epochs shape:        ", self.dwt_epochs.shape)
            print("CSP epochs shape:        ", self.csp_epochs.shape)
            print("Labels shape:            ", self.labels.shape)
            print("Subject IDs shape:       ", self.subject_ids.shape)
            print(Colors.ENDC)

    def epoch_recording(
        self,
        path: Path,
        run_labels: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads one EDF run and slices out fixed-length movement epochs.

        Returns the 4 arrays below.
        1. raw_epochs : np.ndarray, (N, T_raw, num_channels)
        2. bp_epochs  : np.ndarray, (N, T_bp, num_channels * 6)
        3. dwt_epochs : np.ndarray, (N, num_channels * 6)
        4. labels     : np.ndarray, (N,)
        """
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

        if raw.info["sfreq"] != 160:
            print(f"Skipping {path.name}: sfreq={raw.info['sfreq']}")
            return np.array([]), np.array([]), np.array([]), np.array([])

        raw.rename_channels(lambda name: name.rstrip(".").upper())
        sfreq = raw.info["sfreq"]

        events, event_ids = mne.events_from_annotations(raw, verbose=False)
        event_codes = {
            event_ids[name]: label
            for name, label in run_labels.items()
            if name in event_ids
        }

        raw._data *= 1e6  # volts -> uV

        raw.filter(l_freq=4, h_freq=50, verbose=False)
        raw.notch_filter(freqs=60, verbose=False)

        # --- ICA artifact removal --------------------------------------------

        ica_raw = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

        ica = mne.preprocessing.ICA(
            n_components=20,
            method="picard",
            random_state=42,
            max_iter="auto",
            verbose=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            ica.fit(ica_raw, verbose=False)

            eog_component_idx, _ = ica.find_bads_eog(
                ica_raw, ch_name=["FP1", "FP2"], verbose=False
            )
            ica.exclude = eog_component_idx
            raw = ica.apply(raw, verbose=False)

        raw.set_eeg_reference("average", projection=False, verbose=False)

        filtered: NDArray = raw.get_data().T  # (T, num_channels)

        bp_features = compute_bandpower_features(
            filtered,
            sfreq=sfreq,
            window_sec=self.bp_window_sec,
            step_samples_128=self.bp_step_samples,
        )  # (T_bp, num_channels * 6)
        bp_rate = sfreq / self.bp_step_samples  # bandpower timesteps per second
        nperseg = int(
            self.bp_window_sec * sfreq
        )  # number of raw samples in one bandpower window
        bp_offset = (
            (nperseg // 2) / sfreq
        )  # time of the first bandpower timestep relative to the start of the raw signal
        bp_times = (
            bp_offset + np.arange(len(bp_features)) / bp_rate
        )  # time of each bandpower timestep relative to the start of the raw signal

        # --- slice one epoch per T1/T2 movement annotation -------------------

        t_raw = round(
            self.motor_exec_sec * sfreq
        )  # number of raw samples in a 3s epoch
        t_bp = (
            round((self.motor_exec_sec - self.bp_window_sec) * bp_rate) + 1
        )  # number of bandpower timesteps in a 3s epoch

        raw_epochs, bp_epochs, dwt_epochs, labels = [], [], [], []

        for i, event in enumerate(events):
            # check that this event is a T1/T2 movement annotation, not a rest period
            if event[2] not in event_codes:
                continue

            # event[0] is the sample index of the event
            # raw.first_samp is the sample index of the first sample in the raw data
            cls = event_codes[event[2]]
            onset_raw = event[0] - raw.first_samp + round(self.epoch_start_sec * sfreq)
            onset_time = onset_raw / sfreq
            onset_bp = int(np.searchsorted(bp_times, onset_time))

            # T0 marks the next rest period. skip any epoch that would extend
            # into rest or beyond the end of this run.
            if i + 1 < len(events):
                next_event = events[i + 1, 0] - raw.first_samp
            else:
                next_event = len(filtered)

            if (
                onset_raw + t_raw > next_event
                or onset_raw + t_raw > len(filtered)
                or onset_bp + t_bp > len(bp_features)
            ):
                continue  # trial got cut off at the end of the recording

            raw_epoch = filtered[onset_raw : onset_raw + t_raw]

            # --- DWT feature extraction, one call per epoch ------------------

            dwt_epoch = compute_dwt_features(raw_epoch)

            raw_epochs.append(raw_epoch)
            dwt_epochs.append(dwt_epoch)
            bp_epochs.append(bp_features[onset_bp : onset_bp + t_bp])
            labels.append(cls)

        if not labels:
            return np.array([]), np.array([]), np.array([]), np.array([])

        return (
            np.stack(raw_epochs),
            np.stack(bp_epochs),
            np.stack(dwt_epochs),
            np.array(labels),
        )

    def __len__(self) -> int:
        return len(self.train_idx) if self.mode == "train" else len(self.val_idx)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the raw-channel epoch, bandpower epoch, CSP epoch, DWT epoch,
        and gesture-class label at the given index from the training or
        validation split.
        """
        split_idx = self.train_idx if self.mode == "train" else self.val_idx
        i = split_idx[index]

        return (
            torch.tensor(self.raw_epochs[i]),
            torch.tensor(self.bp_epochs[i]),
            torch.tensor(self.csp_epochs[i]),
            torch.tensor(self.dwt_epochs[i]),
            torch.tensor(self.labels[i]),
        )

    def get_split(self, mode: str) -> "PhysioNetGestureDataset":
        split_dataset = object.__new__(PhysioNetGestureDataset)
        split_dataset.__dict__ = self.__dict__.copy()
        split_dataset.mode = mode
        return split_dataset

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

        return sample_weights, class_weights_tensor  # (N,) and (num_classes,)
