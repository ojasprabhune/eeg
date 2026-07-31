from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from eeg.big_hand.position_llm import RegionTokenizer
from eeg.big_hand.position_llm.vqvae import VQVAE
from eeg.data_collection import JointData

from .utils import Colors, appendages, compute_bandpower_features


class TemporalDataset(Dataset):
    def __init__(
        self,
        eeg_data_path: str = "/var/log/thavamount/eeg_dataset/home_eeg",
        hand_data_path: str = "/var/log/thavamount/eeg_dataset/hand_data",
        vqvae_path: str = "/var/log/thavamount/eeg_ckpts/eeg_vqvae/vqvae_final_1250.pth",
        region_tokenizer_path: str = "models/appendages",
        seq_len: int = 60,
        stride: int = 15,
        device: str = "cpu",
        mode: str = "train",
        data_mode: str = "bp",
        val_ratio: float = 0.2,
        train_fraction: float = 1.0,
        verbose: bool = False,
    ) -> None:
        """
        Dataset for regressing EEG data to hand movements. Loads and
        preprocesses the data.
        """

        print(f"{Colors.HEADER}{Colors.BOLD}Initializing dataset...{Colors.ENDC}")
        self.verbose = verbose
        self.seq_len = seq_len
        self.stride = stride
        self.device = device
        self.data_mode = data_mode

        super().__init__()

        # --- vqvae -----------------------------------------------------------

        print(f"{Colors.OKBLUE}Getting VQVAE model...{Colors.ENDC}")
        self.vqvae = VQVAE(input_dim=12, codebook_size=512, embedding_dim=1024)
        vqvae_state_dict = torch.load(vqvae_path, map_location=device)
        self.vqvae.load_state_dict(vqvae_state_dict["model"])
        self.vqvae.to(device)
        self.vqvae.eval()

        self.region_tokenizer = RegionTokenizer(region_tokenizer_path)

        # --- EEG -------------------------------------------------------------

        print(f"{Colors.OKBLUE}Getting EEG data...{Colors.ENDC}")
        self.raws = []
        for path in sorted(Path(f"{eeg_data_path}").rglob("*_cut_raw.fif")):
            raw = mne.io.read_raw_fif(path, verbose=False)
            # some edf+ files have 4 fewer channels
            if raw.info["nchan"] == 71:
                extra_channels = [
                    "OrTimestampS",
                    "OrTimestampMs",
                    "MOT.OrTimestampS",
                    "MOT.OrTimestampM",
                ]
                raw.drop_channels(extra_channels)
            self.raws.append(raw)

        # - filtering -
        self.raw = mne.concatenate_raws(self.raws, preload=True, verbose=False)
        if isinstance(self.raw, tuple):
            self.raw = self.raw[0]
        self.filtered = self.raw.copy().filter(l_freq=0.1, h_freq=50, verbose=False)
        self.filtered = self.filtered.notch_filter(freqs=60, verbose=False)
        self.filtered.set_eeg_reference(
            "average", projection=False, verbose=False
        )  # common average reference

        # - pick channels before branching -
        self.eeg_channels = [
            "AF3",
            "F7",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4",
        ]
        self.filtered.pick(self.eeg_channels, verbose=False)

        # --- 128 Hz branch: bandpower features -------------------------------
        # keep a copy at native rate BEFORE resampling

        self.sfreq_native = self.filtered.info["sfreq"]  # should be 128
        eeg_128hz = self.filtered.get_data().T  # (T_128, 14)

        print(
            f"{Colors.OKBLUE}Computing bandpower features at {self.sfreq_native} Hz...{Colors.ENDC}"
        )
        step_samples_128 = 4
        self.bandpower_features_raw = compute_bandpower_features(
            eeg_128hz,
            sfreq=self.sfreq_native,
            window_sec=1.0,
            step_samples_128=step_samples_128,  # ≈ 32 Hz output
        )

        print(
            f"{Colors.OKGREEN}Bandpower features shape (32 Hz): "
            f"{self.bandpower_features_raw.shape}{Colors.ENDC}"
        )

        # --- 30 Hz branch: raw EEG filtering ---------------------------------

        filtered_30 = self.filtered.copy()
        filtered_30.filter(
            8, 30, method="iir", iir_params=dict(order=4, ftype="butter"), verbose=False
        )
        filtered_30.resample(sfreq=29.973234, verbose=False)

        self.eeg_data: np.ndarray = filtered_30.get_data().T  # (T_30, 14)

        print(f"{Colors.OKGREEN}Filtered & processed EEG data.{Colors.ENDC}")

        # --- labels ----------------------------------------------------------

        print(f"{Colors.OKBLUE}Getting labels...{Colors.ENDC}")
        self.label_files = []
        for path in sorted(Path(f"{hand_data_path}").rglob("*labels_cut.npy")):
            self.label_files.append(np.load(path))

        self.labels = (np.concatenate(self.label_files, axis=0) - 1).astype(np.int64)

        # --- appendages + regions --------------------------------------------

        print(f"{Colors.OKBLUE}Getting appendage data...{Colors.ENDC}")
        self.hands = []
        for path in sorted(Path(f"{hand_data_path}").rglob("*hands_cut.npy")):
            self.hands.append(np.load(path))

        self.raw_app_data = np.concatenate(self.hands, axis=1)  # along time dim
        self.data_joints = JointData(self.raw_app_data)
        self.app_data = appendages(self.data_joints)  # (T, 12)
        self.app_data = np.array(self.region_tokenizer.scaler.transform(self.app_data))

        print(f"{Colors.OKGREEN}Retrieved appendage data.{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Successful retrieved all data.{Colors.ENDC}")

        # --- align bandpower to 30 Hz time grid (interpolation) --------------

        bp_rate = self.sfreq_native / step_samples_128  # 32 Hz
        window_sec = 1.0
        nperseg = int(window_sec * self.sfreq_native)
        bp_offset = (nperseg // 2) / self.sfreq_native
        bp_times = bp_offset + np.arange(len(self.bandpower_features_raw)) / bp_rate

        target_sfreq = filtered_30.info["sfreq"]
        target_len = min(len(self.eeg_data), len(self.app_data), len(self.labels))
        target_times = np.arange(target_len) / target_sfreq

        valid = target_times <= bp_times[-1]
        target_times = target_times[valid]
        target_len = len(target_times)

        bp_aligned = np.zeros(
            (target_len, self.bandpower_features_raw.shape[1]), dtype=np.float32
        )
        for f in range(self.bandpower_features_raw.shape[1]):
            bp_aligned[:, f] = np.interp(
                target_times, bp_times, self.bandpower_features_raw[:, f]
            )

        self.bandpower_features = bp_aligned
        self.eeg_data = self.eeg_data[:target_len]
        self.app_data = self.app_data[:target_len]
        self.labels = self.labels[:target_len]
        min_len = target_len

        print(
            f"{Colors.OKGREEN}Aligned all streams to {target_sfreq:.2f} Hz, "
            f"{min_len} samples{Colors.ENDC}"
        )

        # - vq-vae pre-computing -
        print(f"{Colors.OKBLUE}Pre-computing VQ-VAE tokens...{Colors.ENDC}")
        self.vqvae_tokens_all = []
        chunk_size = 2048
        with torch.no_grad():
            for i in tqdm(range(0, len(self.app_data), chunk_size)):
                chunk = self.app_data[i : i + chunk_size, :]
                chunk_tensor = (
                    torch.tensor(chunk, dtype=torch.float32).to(device).unsqueeze(0)
                )
                tokens = self.vqvae.encode(chunk_tensor)
                self.vqvae_tokens_all.append(tokens.cpu().numpy().flatten())
        self.vqvae_tokens_all = np.concatenate(self.vqvae_tokens_all)

        if verbose:
            print(Colors.HEADER)
            print("EEG shape (30Hz):   ", self.eeg_data.shape)
            print("Bandpower shape:    ", self.bandpower_features.shape)
            print("Appendages shape:   ", self.app_data.shape)
            print("VQ-VAE tokens shape:", self.vqvae_tokens_all.shape)
            print("Labels shape:       ", self.labels.shape)
            print(Colors.ENDC)

        # --- sequences -------------------------------------------------------

        all_eeg, all_bp, all_app, all_tokens, all_labels = [], [], [], [], []

        # we iterate over the data in steps of stride, creating chunks of length
        # seq_len. for each chunk, we extract the corresponding segments of
        # eeg_data, bandpower_features, app_data, vqvae_tokens_all, and labels,
        # and store them in lists. this way we create overlapping sequences of
        # data that can be used for training a temporal model. the resulting
        # chunks will have shape (num_chunks, seq_len, feature_dim) for eeg,
        # bandpower, and appendage data, and (num_chunks, seq_len) for tokens
        # and labels
        for i in range(0, min_len - seq_len + 1, stride):
            all_eeg.append(self.eeg_data[i : i + seq_len, :])
            all_bp.append(self.bandpower_features[i : i + seq_len, :])
            all_app.append(self.app_data[i : i + seq_len, :])
            all_tokens.append(self.vqvae_tokens_all[i : i + seq_len])
            all_labels.append(self.labels[i : i + seq_len])

        self.eeg_chunks = np.array(all_eeg, dtype=np.float32)
        self.bp_chunks = np.array(all_bp, dtype=np.float32)
        self.app_chunks = np.array(all_app, dtype=np.float32)
        self.token_chunks = np.array(all_tokens, dtype=np.int64)
        self.label_chunks = np.array(all_labels, dtype=np.int64)

        # --- train-val split (large contiguous blocks, no leakage) ----------

        n_chunks = len(self.bp_chunks)
        block_size_samples = 9000  # ~5 min at 30 Hz
        n_blocks = min_len // block_size_samples

        block_labels = []
        for b in range(n_blocks):
            s = b * block_size_samples
            e = s + block_size_samples
            block_labels.append(int(np.bincount(self.labels[s:e]).argmax()))
        block_labels = np.array(block_labels)

        rng = np.random.RandomState(42)
        block_assignment = np.full(n_blocks, -1, dtype=np.int8)
        for cls in range(4):
            cls_ids = np.where(block_labels == cls)[0]
            rng.shuffle(cls_ids)
            n_val = max(1, int(len(cls_ids) * val_ratio))
            block_assignment[cls_ids[:n_val]] = 1  # val
            block_assignment[cls_ids[n_val:]] = 0  # train

        sample_assignment = np.full(min_len, -1, dtype=np.int8)
        for b in range(n_blocks):
            s = b * block_size_samples
            e = s + block_size_samples
            sample_assignment[s:e] = block_assignment[b]

        train_idx, val_idx = [], []
        for chunk_i, start in enumerate(range(0, min_len - seq_len + 1, stride)):
            assignments = sample_assignment[start : start + seq_len]
            if (assignments == 0).all():
                train_idx.append(chunk_i)
            elif (assignments == 1).all():
                val_idx.append(chunk_i)

        train_idx = np.array(train_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)

        # --- Z-score bandpower (train stats only) ---------------------------

        train_bp_flat = self.bp_chunks[train_idx].reshape(-1, 84)
        self.bp_mean = train_bp_flat.mean(axis=0, keepdims=True)
        self.bp_std = train_bp_flat.std(axis=0, keepdims=True) + 1e-8

        self.bp_chunks = (
            ((self.bp_chunks.reshape(-1, 84) - self.bp_mean) / self.bp_std)
            .reshape(self.bp_chunks.shape)
            .astype(np.float32)
        )

        # --- subsample training data for scaling experiments ----------------

        if train_fraction < 1.0:
            rng_sub = np.random.RandomState(123)
            chunk_labels_train = np.array(
                [
                    np.bincount(self.label_chunks[i], minlength=4).argmax()
                    for i in train_idx
                ]
            )
            keep_indices = []
            for cls in range(4):
                cls_indices = np.where(chunk_labels_train == cls)[0]
                n_keep = max(1, int(len(cls_indices) * train_fraction))
                rng_sub.shuffle(cls_indices)
                keep_indices.extend(train_idx[cls_indices[:n_keep]])
            train_idx = np.sort(np.array(keep_indices, dtype=np.int64))
            print(
                f"{Colors.WARNING}Scaling: using {train_fraction:.0%} of training data "
                f"({len(train_idx)} chunks){Colors.ENDC}"
            )

        # --- select split ---------------------------------------------------

        if mode == "train":
            split_idx = train_idx
        else:
            split_idx = val_idx

        n_discarded = n_chunks - len(train_idx) - len(val_idx)
        print(
            f"{Colors.OKBLUE}Split: {len(train_idx)} train, "
            f"{len(val_idx)} val, "
            f"{n_discarded} discarded (boundary){Colors.ENDC}\n"
        )

        self.eeg_chunks_split = self.eeg_chunks[split_idx]
        self.bp_chunks_split = self.bp_chunks[split_idx]
        self.app_chunks_split = self.app_chunks[split_idx]
        self.token_chunks_split = self.token_chunks[split_idx]
        self.label_chunks_split = self.label_chunks[split_idx]

        if verbose:
            chunk_labels = np.array(
                [np.bincount(c).argmax() for c in self.label_chunks_split]
            )
            for cls, name in enumerate(["Fist", "Left", "Fingers", "Open"]):
                pct = (chunk_labels == cls).mean()
                print(f"{name} chunks: {pct:.1%}")
            print()

    def __len__(self) -> int:
        return len(self.bp_chunks_split)

    def __getitem__(
        self, index: int | float
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns EEG, bandpower, appendage data, VQ-VAE tokens, labels, durations, and
        masks for the given index from the training set.
        """

        # ensure index is a plain integer for numpy
        if torch.is_tensor(index):
            index = index.item()
        index = int(index)

        eeg = self.eeg_chunks_split[index]
        bp = self.bp_chunks_split[index]
        apps = self.app_chunks_split[index]
        tokens = self.token_chunks_split[index]
        labels = self.label_chunks_split[index]

        chunk_label = np.bincount(labels).argmax()

        # vqvae_tokens: (T,)
        reversed_tokens = tokens[::-1]
        durations = [1]
        for i in range(1, len(reversed_tokens)):
            current_tok = reversed_tokens[i]
            prev_tok = reversed_tokens[i - 1]

            if prev_tok == current_tok:
                durations.append(durations[-1] + 1)
            else:
                durations.append(1)

        durations = durations[::-1]

        masks: list[int | float] = [1]
        for i in range(len(tokens) - 1):
            current_tok = tokens[i]
            next_tok = tokens[i + 1]

            if next_tok == current_tok:
                masks.append(1e-4)
            else:
                masks.append(1)

        return (
            torch.tensor(eeg),
            torch.tensor(bp),
            torch.tensor(apps),
            torch.tensor(tokens),
            torch.tensor(chunk_label),
            torch.tensor(durations),
            torch.tensor(masks),
        )

    def get_sampler_weights(self) -> tuple[list[float], torch.Tensor]:
        # self.label_chunks is (N, seq_len). we want the label that defines the
        # chunk. find the majority label of the sequence (the mode).
        chunk_labels = np.array(
            [np.bincount(c).argmax() for c in self.label_chunks_split]
        )

        # force a fixed class count length so absent classes are handled safely.
        num_classes = 4
        class_counts = np.bincount(chunk_labels, minlength=num_classes)

        # calculate weight per class: total / (num_classes * class count).
        # add a tiny epsilon to avoid division by zero if a class isn't present.
        total_samples = len(chunk_labels)
        weights_per_class = total_samples / (num_classes * (class_counts + 1e-8))

        # assign the specific weight to every individual sample
        sample_weights = [
            float(weights_per_class[int(label)]) for label in chunk_labels
        ]

        # for CrossEntropyLoss we need a fixed-length weight tensor
        class_weights_tensor = torch.tensor(weights_per_class, dtype=torch.float32)

        return sample_weights, class_weights_tensor
