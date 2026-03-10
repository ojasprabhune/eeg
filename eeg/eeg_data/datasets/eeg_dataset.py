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
        use_vqvae: bool = True,
        device: str = "cpu",
        print_shapes: bool = False,
    ) -> None:
        """
        Dataset for regressing EEG data to hand movements. Loads and
        preprocesses the data.
        """

        print(f"{Colors.HEADER}{Colors.BOLD}Initializing dataset...{Colors.ENDC}")

        super().__init__() 

        # --- vqvae ---

        print(f"{Colors.OKBLUE}Getting VQVAE model...{Colors.ENDC}")
        self.vqvae = VQVAE(input_dim=12, codebook_size=512, embedding_dim=1024)
        vqvae_state_dict = torch.load(vqvae_path, map_location=device)
        self.vqvae.load_state_dict(vqvae_state_dict["model"])
        self.vqvae.to(device)
        self.vqvae.eval()
        self.use_vqvae = use_vqvae

        self.region_tokenizer = RegionTokenizer(region_tokenizer_path)

        # --- EEG ---

        print(f"{Colors.OKBLUE}Getting EEG data...{Colors.ENDC}")
        self.raws = []
        for path in Path(f"{eeg_data_path}").rglob("*_cut_raw.fif"):
            self.raws.append(mne.io.read_raw_fif(path))

        # - filtering -
        self.raw: mne.io.Raw = mne.concatenate_raws(self.raws, preload=True)
        self.filtered = self.raw.copy().filter(l_freq=0.1, h_freq=50)  # band-pass filter
        self.filtered = self.filtered.notch_filter(freqs=60)  # notch filtering
        self.filtered.set_eeg_reference("average", projection=False)  # common average reference
        self.filtered.filter(8, 30, method="iir", iir_params=dict(order=4, ftype="butter"))  # butterworth

        # - processing -
        self.eeg_channels = [
            "AF3","F7","F3","FC5","T7","P7","O1",
            "O2","P8","T8","FC6","F4","F8","AF4"
        ]
        self.filtered.pick(self.eeg_channels)

        # TODO use accel + mag data & remove low EEG quality segments
        # TODO fix sampling frequencies for EEG and hand if required

        self.eeg_data: np.ndarray = self.filtered.get_data()  # (C, T)
        
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
        self.region_tokens = self.region_tokenizer.encode(torch.tensor(self.app_data))  # (T,)

        print(f"{Colors.OKGREEN}Retrieved appendage data.{Colors.ENDC}")

        if print_shapes:
            print("EEG shape:       ", self.eeg_data.shape) # (14, T)
            print("raw app shape:   ", self.raw_app_data.shape) # (2, T, 63)
            print("app shape:       ", self.app_data.shape) # (T, 12)
            print("regions shape:   ", self.region_tokens.shape) # (T,)

        print(f"{Colors.OKGREEN}Successful retrieved all data.{Colors.ENDC}")

        # - vq-vae pre-computing -
        self.vqvae_tokens_all = []
        if self.use_vqvae:
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
                print("VQ-VAE tokens shape:", self.vqvae_tokens_all.shape) # (T,)

        # --- sequences ---
        # self...
        # self...
        # self...
        # self...

        # TODO sequencing

    def __len__(self) -> int:
        return len(self.regions)

    # def __getitem__(self, index: int) -> tuple | dict[str, list[int]]:
    #     """
    #     Called when we do dataset[idx]
    #     """

    #     # returning (1, T) of vqvae tokens to input into PositionLLM
    #     # returning (T, 12) of appendage values
    #     # TODO add eegs to this if applicable
    #     if self.use_vqvae:
    #         vqvae_tokens = self.vqvae_token_crops[index]

    #         if not self.duration_prediction:
    #             return vqvae_tokens, self.appendages[index], self.eegs[index]
    #         else:
    #             # vqvae_tokens, appendage_values, durations, mask
    #             # vqvae_tokens: (T,)
    #             reversed_vqvae_toks = vqvae_tokens[::-1]
    #             durations = [1]
    #             for i in range(1, len(reversed_vqvae_toks)):
    #                 # counting backwards
    #                 current_tok = reversed_vqvae_toks[i]
    #                 prev_tok = reversed_vqvae_toks[i - 1]

    #                 if prev_tok == current_tok:
    #                     durations.append(durations[-1] + 1)
    #                 else:
    #                     durations.append(1)

    #             durations = durations[::-1]

    #             masks = [1]
    #             for i in range(len(vqvae_tokens) - 1):
    #                 current_tok = vqvae_tokens[i]
    #                 next_tok = vqvae_tokens[i + 1]

    #                 if next_tok == current_tok:
    #                     masks.append(0)
    #                 else:
    #                     masks.append(1)

    #             return {
    #                 "tokens": vqvae_tokens,
    #                 "values": self.appendages[index],
    #                 "durations": durations,
    #                 "masks": masks,
    #                 "eegs": self.eegs[index],
    #             }

    #     else:
    #         return self.regions[index], self.appendages[index], self.eegs[index]

    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     Collate function to be used with DataLoader to batch data.
    #     """

    #     if isinstance(batch[0], dict):
    #         region_tokens = torch.tensor(
    #             [item["tokens"] for item in batch]).to(torch.int64)  # (B, T)
    #         appendage_values = torch.tensor(
    #             # (B, T, 12)
    #             [item["values"] for item in batch]).to(torch.float32)
    #         durations = torch.tensor([item["durations"] for item in batch]).to(
    #             torch.float32)  # (B, T)
    #         masks = torch.tensor([item["masks"]
    #                              # (B, T)
    #                               for item in batch]).to(torch.float32)
    #         eegs = torch.tensor([item["eegs"] for item in batch]).to(
    #             torch.float32)  # (B, T, num_eeg_channels)

    #         return {
    #             "tokens": region_tokens,
    #             "values": appendage_values,
    #             "durations": durations,
    #             "masks": masks,
    #             "eegs": eegs
    #         }

    #     region_tokens = torch.tensor(
    #         [item[0] for item in batch]).to(torch.int64)  # (B, T)
    #     appendage_values = torch.tensor(
    #         [item[1] for item in batch]).to(torch.float32)  # (B, T, 12)
    #     eeg_values = torch.stack(
    #         # (B, T, num_eeg_channels)
    #         [item[2] for item in batch]).to(torch.float32)

    #     return region_tokens, appendage_values, eeg_values
