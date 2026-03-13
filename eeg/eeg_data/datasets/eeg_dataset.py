from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import mne

import torch
import torchaudio
from torch.utils.data import Dataset

from .utils import appendages, Colors
from eeg.data_collection import JointData
from eeg.big_hand.position_llm import RegionTokenizer
from eeg.big_hand.position_llm.vqvae import VQVAE

class EEGDataset(Dataset):
    def __init__(
        self,
        eeg_data_path: str = "/var/log/thavamount/eeg_dataset/home_eeg",
        hand_data_path: str = "/var/log/thavamount/eeg_dataset/hand_data",
        vqvae_path: str = "/var/log/thavamount/eeg_ckpts/eeg_vqvae/vqvae_final_1250.pth",
        region_tokenizer_path: str = "models/appendages",
        seq_len: int = 900,
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
        for path in Path(f"{eeg_data_path}").rglob("*_cut_raw.fif"):
            self.raws.append(mne.io.read_raw_fif(path))

        # - filtering -
        self.raw: mne.io.Raw = mne.concatenate_raws(self.raws, preload=True) # type: ignore
        self.filtered = self.raw.copy().filter(l_freq=0.1, h_freq=50)
        self.filtered = self.filtered.notch_filter(freqs=60)  # type: ignore
        self.filtered.set_eeg_reference("average", projection=False)  # common average reference
        self.filtered.filter(8, 30, method="iir", iir_params=dict(order=4, ftype="butter"))  # butterworth

        # - processing -
        self.eeg_channels = [
            "AF3","F7","F3","FC5","T7","P7","O1",
            "O2","P8","T8","FC6","F4","F8","AF4"
        ]
        self.filtered.pick(self.eeg_channels)
        self.filtered.resample(sfreq=29.973234)

        # TODO use accel + mag data & remove low EEG quality segments
        # TODO fix sampling frequencies for EEG and hand if required

        # (C, T)
        self.eeg_data: np.ndarray = self.filtered.get_data().T # type: ignore
        
        print(f"{Colors.OKGREEN}Filtered & processed EEG data.")

        # --- appendages + regions ---

        print(f"{Colors.OKBLUE}Getting appendage data...{Colors.ENDC}")
        self.hands = []
        for path in Path(f"{hand_data_path}").rglob("*_cut.npy"):
            self.hands.append(np.load(path))

        self.raw_app_data = np.concatenate(self.hands, axis=1) # along time dim
        self.data_joints = JointData(self.raw_app_data)
        self.app_data = appendages(self.data_joints)  # (T, 12)
        self.app_data = self.region_tokenizer.scaler.transform(self.app_data)

        print(f"{Colors.OKGREEN}Retrieved appendage data.{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Successful retrieved all data.{Colors.ENDC}")

        # - vq-vae pre-computing -
        self.vqvae_tokens_all = []
        print(f"{Colors.OKBLUE}Pre-computing VQ-VAE tokens...{Colors.ENDC}")
        self.vqvae_tokens_all = []
        chunk_size = 2048  # process in chunks
        with torch.no_grad():
            for i in tqdm(range(0, len(self.app_data), chunk_size)):
                chunk = self.app_data[i: i + chunk_size, :]
                chunk_tensor = (
                    torch.tensor(chunk, dtype=torch.float32)
                    .to(device)
                    .unsqueeze(0)
                )
                tokens = self.vqvae.encode(chunk_tensor)
                self.vqvae_tokens_all.append(tokens.cpu().numpy().flatten())
        self.vqvae_tokens_all = np.concatenate(self.vqvae_tokens_all)

        if print_shapes:
            print("EEG shape:          ", self.eeg_data.shape) # (14, T)
            print("app shape:          ", self.app_data.shape) # (T, 12)
            print("VQ-VAE tokens shape:", self.vqvae_tokens_all.shape) # (T,)

        # --- sequences ---
        self.eeg_chunks = []
        self.app_chunks = []
        self.token_chunks = []

        # all chunks are length seq_len
        for i in range(0, len(self.eeg_data), seq_len):
            eeg_chunk = self.eeg_data[i : i + seq_len, :]  # shape: (seq_len, 14)
            app_chunk = self.app_data[i : i + seq_len, :]  # shape: (seq_len, 12)
            token_chunk = self.vqvae_tokens_all[i : i + seq_len]  # shape: (seq_len,)

            if eeg_chunk.shape[0] == seq_len:
                self.eeg_chunks.append(eeg_chunk)
                self.app_chunks.append(app_chunk)
                self.token_chunks.append(token_chunk)

        # --- train-val split ---

        # index at 80% on time dim
        self.split_idx = int(len(self.eeg_chunks) * 0.8)

        self.eeg_chunks = np.array(self.eeg_chunks, dtype=np.float32)
        self.app_chunks = np.array(self.app_chunks, dtype=np.float32)
        self.token_chunks = np.array(self.token_chunks, dtype=np.int64)

        self.train_eeg_chunks = self.eeg_chunks[:self.split_idx, :, :]
        self.train_app_chunks = self.app_chunks[:self.split_idx, :, :]
        self.train_token_chunks = self.token_chunks[:self.split_idx]

        self.val_eeg_chunks = self.eeg_chunks[self.split_idx:, :, :]
        self.val_app_chunks = self.app_chunks[self.split_idx:, :, :]
        self.val_token_chunks = self.token_chunks[self.split_idx:]
        
        if print_shapes:
            print(f"{Colors.WARNING}total # of chunks: {self.__len__()}{Colors.ENDC}")

    def __len__(self) -> int:
        return len(self.train_eeg_chunks)

    def __getitem__(self, index: int) -> tuple[list[list[int]], list[list[int]], list[int], list[int], list[int]]:
        """
        Returns the EEG data, appendage data, and VQ-VAE tokens for the given
        index from the training set.
        """

        eeg: list[list[int]] = self.train_eeg_chunks[index]
        apps: list[list[int]] = self.train_app_chunks[index]
        tokens: list[int] = self.train_token_chunks[index]

        # vqvae_tokens: (T,)
        reversed_tokens = tokens[::-1]
        durations = [1]
        for i in range(1, len(reversed_tokens)):
            # counting backwards
            current_tok = reversed_tokens[i]
            prev_tok = reversed_tokens[i - 1]

            if prev_tok == current_tok:
                durations.append(durations[-1] + 1)
            else:
                durations.append(1)
        
        durations = durations[::-1]

        masks = [1]
        for i in range(len(tokens) - 1):
            current_tok = tokens[i]
            next_tok = tokens[i + 1]

            if next_tok == current_tok:
                masks.append(0)
            else:
                masks.append(1)

        return eeg, apps, tokens, durations, masks

    def get_val_data(self, index: int) -> tuple[list[list[int]], list[list[int]], list[int]]:
        """
        Returns the EEG data, appendage data, and VQ-VAE tokens for the given
        index from the validation set.
        """

        eeg: list[list[int]] = self.val_eeg_chunks[index]
        apps: list[list[int]] = self.val_app_chunks[index]
        tokens: list[int] = self.val_token_chunks[index]

        return eeg, apps, tokens
