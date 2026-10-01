import re
import warnings
from pathlib import Path

import mne
import numpy as np
import pyxdf
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ..utils import gesture_experiments
from .utils import (
    EMOTIV_CHANNELS,
    Colors,
    compute_bandpower_features,
    compute_dwt_features,
)


class GestureDataset(Dataset):
    """
    Dataset to load raw LSL and EmotivPRO recordings as .xdf and slices out
    epochs for motor execution trials. Each epoch is labeled with the gesture
    class that was cued for it. It does preprocessing like ICA and low-, high-,
    and bandpass filters on the EEG. We do CSP, DWT, and bandpower feature
    extraction and return all four types (including raw) and labels.

    See notebooks/lsl.ipynb for the event/epoch extraction this is based on,
    and scripts/data/eeg_scheduling/cueing.py for how markers get written
    during recording: REST_START, CUE_{class}, PRE_{class}, MOVE_{class},
    END_SEQ, SESSION_END, where {class} is the 1-indexed gesture class.
    """

    def __init__(
        self,
        experiment: str,
        recordings_path: str = "/Users/ojasprabhune/Documents/research/NORA/recordings/lsl/sub-P001/ses-S002/",
        mode: str = "train",
        k: int = 1,
        fold: int = 0,
        motor_exec_sec: float = 3.0,
        bp_window_sec: float = 1.0,
        bp_step_samples: int = 4,
        verbose: bool = False,
    ) -> None:
        """
        Input is every recorded .xdf session under recordings_path. Each
        MOVE_{class} marker becomes one epoch spanning motor_exec_sec seconds
        (matching cueing.py's MOTOR_EXEC), labeled with class - 1.

        k/fold pick one of k stratified cross-validation folds: fold's
        examples become val, the other k-1 folds become train. We call this
        constructor once per fold (same k, fold=0..k-1) to run full k-fold CV.
        """

        print(
            f"{Colors.HEADER}{Colors.BOLD}Initializing gesture dataset...{Colors.ENDC}"
        )
        self.verbose = verbose
        self.mode = mode
        self.motor_exec_sec = motor_exec_sec

        super().__init__()

        self.num_classes = max(gesture_experiments[experiment].values())

        # --- load + epoch every session --------------------------------------

        print(f"{Colors.OKBLUE}Getting recordings...{Colors.ENDC}")

        session_path = sorted(Path(recordings_path).rglob("*_eeg.xdf"))
        if len(session_path) != 1:
            raise ValueError("Multiple files are in session path.")

        raw_epochs, bp_epochs, dwt_epochs, labels = self.epoch_session(
            session_path[0],
            bp_window_sec=bp_window_sec,
            bp_step_samples=bp_step_samples,
        )

        self.raw_epochs = np.array(raw_epochs).astype(np.float32)
        self.bp_epochs = np.array(bp_epochs).astype(np.float32)
        self.dwt_epochs = np.array(dwt_epochs).astype(np.float32)
        self.labels = np.array(labels).astype(np.int64)

        # raw: (N, T_raw, 14)
        # bp: (N, T_bp, 84)
        # dwt: (N, 84)
        # labels: (N,)

        print(f"{Colors.OKGREEN}Loaded {len(self.labels)} epochs.{Colors.ENDC}")

        if verbose:
            print(Colors.HEADER)
            print("Raw epochs shape:       ", self.raw_epochs.shape)
            print("Bandpower epochs shape: ", self.bp_epochs.shape)
            print("DWT epochs shape:       ", self.dwt_epochs.shape)
            print("Labels shape:           ", self.labels.shape)
            print(Colors.ENDC)

        # --- stratified k-fold split --------------------------------------

        # k=1 means "no split" — every example goes in both train_idx and
        # val_idx, since np.array_split(cls_idx, 1) would otherwise leave
        # val_idx with everything and train_idx with nothing (an empty
        # np.concatenate, which raises). Use this while not doing k-fold CV.
        if k == 1:
            rng = np.random.RandomState(42)
            train_idx, val_idx = [], []

            for cls in range(self.num_classes):
                cls_idx = np.where(self.labels == cls)[0]
                rng.shuffle(cls_idx)

                split_idx = round(len(cls_idx) * 0.8)

                train_idx.extend(cls_idx[:split_idx])
                val_idx.extend(cls_idx[split_idx:])

            self.train_idx = np.array(sorted(train_idx), dtype=np.int64)
            self.val_idx = np.array(sorted(val_idx), dtype=np.int64)

            print(
                f"{Colors.OKBLUE}{len(self.train_idx)} train, "
                f"{len(self.val_idx)} val{Colors.ENDC}\n"
            )

        else:
            # each epoch is an isolated trial with rest periods on either side,
            # so unlike TemporalDataset's sliding windows there's no adjacency
            # leakage to worry about — a plain per-class random split is enough.
            # same shuffle every time (fixed seed) so fold 0..k-1 partition the
            # data identically across separate GestureDataset() calls/processes.
            rng = np.random.RandomState(42)
            train_idx, val_idx = [], []
            for cls in range(self.num_classes):
                cls_idx = np.where(self.labels == cls)[0]
                rng.shuffle(cls_idx)
                # np.array_split handles class counts that don't divide evenly
                # by k (e.g. 52 examples / 5 folds -> folds of size 11,11,10,10,10)
                cls_folds = np.array_split(cls_idx, k)
                val_idx.extend(cls_folds[fold])
                train_idx.extend(
                    np.concatenate([f for i, f in enumerate(cls_folds) if i != fold])
                )

            self.train_idx = np.array(sorted(train_idx), dtype=np.int64)
            self.val_idx = np.array(sorted(val_idx), dtype=np.int64)

            print(
                f"{Colors.OKBLUE}Fold {fold + 1}/{k}: {len(self.train_idx)} train, "
                f"{len(self.val_idx)} val{Colors.ENDC}\n"
            )

        # --- CSP spatial-filter features -------------------------------

        # CSP (Common Spatial Patterns) looks for linear combinations of the
        # 14 electrode channels ("virtual channels") whose variance differs
        # as much as possible between gesture classes. It finds these by
        # solving a generalized eigenvalue problem on the classes'
        # covariance matrices: the resulting eigenvectors ("spatial
        # filters") that come with the largest eigenvalues point in
        # directions where one class's signal power dominates, and the ones
        # with the smallest eigenvalues point in directions where a
        # different class dominates - so keeping filters from both ends of
        # that spectrum captures the most class-discriminative spatial
        # patterns. This is the standard first feature-extraction step in
        # nearly every published EEG motor gesture-decoding pipeline we
        # found in our paper research.
        print(f"{Colors.OKBLUE}Fitting CSP...{Colors.ENDC}")

        # mne.decoding.CSP expects (n_epochs, n_channels, n_times), but
        # raw_epochs is (n_epochs, n_times, n_channels), so swap the last
        # two axes to match
        csp_input = self.raw_epochs.transpose(0, 2, 1).astype(np.float64)

        # reg="ledoit_wolf" shrinks each class's covariance estimate toward
        # a scaled identity matrix before CSP inverts it. With only ~40-50
        # trials per class and 14 channels, the raw covariance estimate is
        # close to singular (inverting it amplifies noise into huge, useless
        # numbers) - shrinkage is the standard fix, pulling the estimate
        # toward something safely invertible at the cost of a small,
        # well-understood bias.
        csp = mne.decoding.CSP(n_components=6, reg="ledoit_wolf", log=True)

        # same reasoning as the RuntimeWarning suppression around ICA above:
        # an internal pseudo-inverse step can warn on divide-by-zero/
        # overflow without producing NaN/Inf in the actual result (verified)
        with mne.utils.use_log_level("error"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # fit only on this fold's training epochs, never validation
            # epochs, so the spatial filters aren't chosen using information
            # from data we'll evaluate the model on later - the same
            # leakage rule that motivates computing the k-fold split above
            # before this point
            csp.fit(csp_input[self.train_idx], self.labels[self.train_idx])

            # transform() applies the fitted spatial filters to every epoch
            # (train and val) and returns, per filter, the log-variance of
            # the resulting virtual channel - one number per component
            # summarizing how much "power" that discriminative direction
            # carried during the trial
            self.csp_epochs = csp.transform(csp_input).astype(np.float32)  # (N, 6)

        if verbose:
            print("CSP epochs shape:       ", self.csp_epochs.shape)

    def epoch_session(
        self, path: Path, bp_window_sec: float, bp_step_samples: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads one .xdf recording and slices it into per-trial epochs.

        Returns
        -------
        raw_epochs : np.ndarray, shape (N, T_raw, 14)
        bp_epochs  : np.ndarray, shape (N, T_bp, 84)
        dwt_epochs : np.ndarray, shape (N, 84)
            Per-trial DWT wavelet-energy features (NEW - see
            compute_dwt_features in datasets/utils.py).
        labels     : np.ndarray, shape (N,)
            Zero-based gesture class per epoch.
        """
        streams, _ = pyxdf.load_xdf(str(path))

        # find specific streams by type
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

        # --- ICA artifact removal ------------------------------------

        # ICA (Independent Component Analysis) treats each of the 14
        # electrode channels as a different mixture (weighted sum) of the
        # same underlying set of source signals - some sources are brain
        # activity, others are things like eye blinks or muscle twitches.
        # ICA finds an "unmixing" matrix that splits the 14 channels back
        # into that many statistically independent components. Eye blinks
        # tend to land cleanly in their own component because they're a
        # large, stereotyped voltage swing that shows up on every channel in
        # a very consistent, distinctive pattern - very different statistics
        # from the underlying brain signal we actually care about.

        # n_components is 13, not 14: set_eeg_reference("average") below
        # removes one degree of freedom from the data (every channel becomes
        # a function of the other 13), so asking ICA to find 14 independent
        # components afterward would be asking for more directions than the
        # data actually has. Running ICA before the average reference avoids
        # that problem.
        n_components = len(EMOTIV_CHANNELS) - 1
        ica = mne.preprocessing.ICA(
            n_components=n_components, random_state=42, max_iter="auto", verbose=False
        )

        # the pseudo-inverse mne computes internally at the end of fit() can
        # raise a harmless "divide by zero"/"overflow" RuntimeWarning when
        # one component's variance is very close to zero - it doesn't affect
        # the actual unmixing result (verified: no NaN/Inf in the output),
        # so it's suppressed here rather than left to print noise every run
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ica.fit(raw, verbose=False)

            # find_bads_eog scores every component by how well its time
            # course correlates with an eye-movement channel and flags the
            # ones that look like blinks. This 14-channel Emotiv headset has
            # no dedicated eye electrode, so we use AF3 - the frontal
            # channel physically closest to the eyes - as a stand-in, a
            # common substitute on consumer EEG rigs without a real EOG
            # channel.
            eog_component_idx, _ = ica.find_bads_eog(raw, ch_name="AF3", verbose=False)

        # tell ICA which components to drop, then reconstruct the 14
        # channels using only the remaining ("clean") components - this is
        # what actually subtracts the blink artifact back out of the signal
        ica.exclude = eog_component_idx
        raw = ica.apply(raw, verbose=False)

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

        raw_epochs, bp_epochs, dwt_epochs, labels = [], [], [], []

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

            raw_epoch = filtered[onset_raw : onset_raw + t_raw]

            raw_epochs.append(raw_epoch)
            bp_epochs.append(bp_features[onset_bp : onset_bp + t_bp])

            # --- DWT feature extraction, one call per epoch ------------

            # see compute_dwt_features in datasets/utils.py for what this
            # actually does; called here (once per trial, on that trial's
            # already-sliced 3s epoch) rather than on the continuous signal
            # since - unlike the sliding-window bandpower features above -
            # DWT decomposes a fixed-length window as a whole, so it fits
            # naturally per-epoch instead of per-timestep.
            dwt_epochs.append(compute_dwt_features(raw_epoch))

            labels.append(cls)

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
