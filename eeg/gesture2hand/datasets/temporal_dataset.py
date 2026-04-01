from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from eeg.big_hand.position_llm import RegionTokenizer
from eeg.big_hand.position_llm.vqvae import VQVAE
from eeg.data_collection import JointData

from .utils import Colors, appendages


def compute_bandpower_features(
    eeg_128hz: np.ndarray,
    sfreq: float = 128.0,
    window_sec: float = 1.0,
    step_samples_128: int = 4,
) -> np.ndarray:
    """
    Compute bandpower features from 128 Hz EEG via FFT.

    Parameters
    ----------
    eeg_128hz : np.ndarray, shape (T_128, 14)
        Filtered EEG at native sample rate.
    sfreq : float
        Sampling frequency.
    window_sec : float
        FFT window length in seconds.
    step_samples_128 : int
        Step size in samples. 4 @ 128 Hz ≈ 32 Hz output.

    Returns
    -------
    features : np.ndarray, shape (T_out, 84)
        14 channels × 6 features (theta, mu, beta, low_gamma, mu/beta, total).
    """
    T, C = eeg_128hz.shape
    nperseg = int(window_sec * sfreq)
    half_win = nperseg // 2

    bands = {
        "theta": (4, 8),
        "mu": (8, 13),
        "beta": (13, 30),
        "low_gamma": (30, 50),
    }

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    print(f"Number of frequencies: {freqs.shape}.")
    band_masks = {
        name: np.logical_and(freqs >= flo, freqs <= fhi)
        for name, (flo, fhi) in bands.items()
    }

    hann = np.hanning(nperseg)[:, None]  # (nperseg, 1)
    centers = np.arange(half_win, T - half_win, step_samples_128)
    n_out = len(centers)
    features = np.zeros((n_out, C * 6), dtype=np.float32)

    for i, t in enumerate(centers):
        segment = eeg_128hz[t - half_win : t + half_win, :] * hann
        fft_vals = np.fft.rfft(segment, axis=0)
        psd = (np.abs(fft_vals) ** 2) / nperseg

        for ch in range(C):
            base = ch * 6
            bp = {}
            for j, (name, mask) in enumerate(band_masks.items()):
                bp[name] = psd[mask, ch].sum()
                features[i, base + j] = bp[name]
            features[i, base + 4] = bp["mu"] / (bp["beta"] + 1e-10)
            features[i, base + 5] = sum(bp.values()) + 1e-10

    return features


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
        print_shapes: bool = False,
    ) -> None:
        """
        Dataset for regressing EEG data to hand movements. Loads and
        preprocesses the data.
        """

        print(f"{Colors.HEADER}{Colors.BOLD}Initializing dataset...{Colors.ENDC}")
        self.print_shapes = print_shapes
        self.device = device
        self.stride = stride

        super().__init__()

        # --- vqvae ---

        print(f"{Colors.OKBLUE}Getting VQVAE model...{Colors.ENDC}")
        self.vqvae = VQVAE(input_dim=12, codebook_size=512, embedding_dim=1024)
        vqvae_state_dict = torch.load(vqvae_path, map_location=device)
        self.vqvae.load_state_dict(vqvae_state_dict["model"])
        self.vqvae.to(device)
        self.vqvae.eval()

        self.region_tokenizer = RegionTokenizer(region_tokenizer_path)

        # --- EEG ---

        print(f"{Colors.OKBLUE}Getting EEG data...{Colors.ENDC}")
        self.raws = []
        for path in sorted(Path(f"{eeg_data_path}").rglob("*_cut_raw.fif")):
            raw = mne.io.read_raw_fif(path, verbose=False)
            # some edf+ files have 4 fewer channels
            if raw.info['nchan'] == 71:
                extra_channels = [
                    "OrTimestampS",
                    "OrTimestampMs",
                    "MOT.OrTimestampS",
                    "MOT.OrTimestampM"
                ]
                raw.drop_channels(extra_channels)
            self.raws.append(raw)

        # - filtering -
        self.raw: mne.io.Raw = mne.concatenate_raws(self.raws, preload=True, verbose=False)
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

        # --- 128 Hz branch: bandpower features ---
        # keep a copy at native rate BEFORE resampling

        self.sfreq_native = self.filtered.info["sfreq"]  # should be 128
        eeg_128hz = self.filtered.get_data().T  # (T_128, 14)

        print(
            f"{Colors.OKBLUE}Computing bandpower features at {self.sfreq_native} Hz...{Colors.ENDC}"
        )
        self.bandpower_features_raw = compute_bandpower_features(
            eeg_128hz,
            sfreq=self.sfreq_native,
            window_sec=1.0,
            step_samples_128=4,  # ≈ 32 Hz output
        )
        # Z-score bandpower features
        bp_mean = self.bandpower_features_raw.mean(axis=0, keepdims=True)
        bp_std = self.bandpower_features_raw.std(axis=0, keepdims=True) + 1e-8
        self.bandpower_features = (
            (self.bandpower_features_raw - bp_mean) / bp_std
        ).astype(np.float32)

        print(
            f"{Colors.OKGREEN}Bandpower features shape: {self.bandpower_features.shape} (T, 84){Colors.ENDC}"
        )

        # --- 30 Hz branch: raw EEG (existing behavior) ---

        filtered_30 = self.filtered.copy()
        filtered_30.filter(
            8, 30, method="iir", iir_params=dict(order=4, ftype="butter"), verbose=False
        )
        filtered_30.resample(sfreq=29.973234, verbose=False)

        self.eeg_data: np.ndarray = filtered_30.get_data().T  # (T_30, 14)

        print(f"{Colors.OKGREEN}Filtered & processed EEG data.{Colors.ENDC}")

        # --- appendages + regions ---

        print(f"{Colors.OKBLUE}Getting appendage data...{Colors.ENDC}")
        self.hands = []
        for path in sorted(Path(f"{hand_data_path}").rglob("*_cut.npy")):
            self.hands.append(np.load(path))

        self.raw_app_data = np.concatenate(self.hands, axis=1)  # along time dim
        self.data_joints = JointData(self.raw_app_data)
        self.app_data = appendages(self.data_joints)  # (T, 12)
        self.app_data = self.region_tokenizer.scaler.transform(self.app_data)

        print(f"{Colors.OKGREEN}Retrieved appendage data.{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Successful retrieved all data.{Colors.ENDC}")

        # ─── align bandpower features with hand data ──────────────────────
        min_len = min(
            len(self.bandpower_features), len(self.app_data), len(self.eeg_data)
        )
        self.bandpower_features = self.bandpower_features[:min_len]
        self.app_data = self.app_data[:min_len]
        self.eeg_data = self.eeg_data[:min_len]

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

        if print_shapes:
            print("EEG shape (30Hz):   ", self.eeg_data.shape)
            print("Bandpower shape:    ", self.bandpower_features.shape)
            print("app shape:          ", self.app_data.shape)
            print("VQ-VAE tokens shape:", self.vqvae_tokens_all.shape)

        # --- sequences ---
        self.eeg_chunks = []
        self.app_chunks = []
        self.token_chunks = []
        self.bp_chunks = []

        for i in range(0, min_len - seq_len + 1, stride):
            eeg_chunk = self.eeg_data[i : i + seq_len, :]  # (seq_len, 14)
            app_chunk = self.app_data[i : i + seq_len, :]  # (seq_len, 12)
            token_chunk = self.vqvae_tokens_all[i : i + seq_len]  # (seq_len,)
            bp_chunk = self.bandpower_features[i : i + seq_len, :]  # (seq_len, 84)

            self.eeg_chunks.append(eeg_chunk)
            self.app_chunks.append(app_chunk)
            self.token_chunks.append(token_chunk)
            self.bp_chunks.append(bp_chunk)

        # --- train-val split ---

        self.split_idx = int(len(self.eeg_chunks) * 0.8)

        self.eeg_chunks = np.array(self.eeg_chunks, dtype=np.float32)
        self.app_chunks = np.array(self.app_chunks, dtype=np.float32)
        self.token_chunks = np.array(self.token_chunks, dtype=np.int64)
        self.bp_chunks = np.array(self.bp_chunks, dtype=np.float32)

        self.train_eeg_chunks = self.eeg_chunks[: self.split_idx, :, :]
        self.train_app_chunks = self.app_chunks[: self.split_idx, :, :]
        self.train_token_chunks = self.token_chunks[: self.split_idx]
        self.train_bp_chunks = self.bp_chunks[: self.split_idx, :, :]

        self.val_eeg_chunks = self.eeg_chunks[self.split_idx :, :, :]
        self.val_app_chunks = self.app_chunks[self.split_idx :, :, :]
        self.val_token_chunks = self.token_chunks[self.split_idx :]
        self.val_bp_chunks = self.bp_chunks[self.split_idx :, :, :]

        if print_shapes:
            print(f"{Colors.WARNING}total # of chunks: {self.__len__()}{Colors.ENDC}")

    def __len__(self) -> int:
        return len(self.train_eeg_chunks)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns EEG, bandpower, appendage data, VQ-VAE tokens, durations, masks
        for the given index from the training set.
        """

        eeg = self.train_eeg_chunks[index]
        bp = self.train_bp_chunks[index]
        apps = self.train_app_chunks[index]
        tokens = self.train_token_chunks[index]

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
            torch.tensor(durations),
            torch.tensor(masks),
        )

    def get_val_data(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns EEG, bandpower, appendage data, and VQ-VAE tokens
        for the given index from the validation set.
        """

        eeg = self.val_eeg_chunks[index]
        bp = self.val_bp_chunks[index]
        apps = self.val_app_chunks[index]
        tokens = self.val_token_chunks[index]

        return (
            torch.tensor(eeg),
            torch.tensor(bp),
            torch.tensor(apps),
            torch.tensor(tokens),
        )
