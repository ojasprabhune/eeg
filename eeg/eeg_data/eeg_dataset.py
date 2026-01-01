from tqdm import tqdm
import numpy as np
import torch
import torchaudio
import mne
from torch.utils.data import Dataset

from .utils import appendages
from eeg.data_collection import JointData
from eeg.big_hand.position_llm import RegionTokenizer
from eeg.big_hand.position_llm.vqvae import VQVAE


class EEGDataset(Dataset):
    def __init__(
        self,
        data_path: str = "/var/log/thavamount/eeg_dataset",
        vqvae_path: str = "/var/log/thavamount/eeg_ckpts/eeg_vqvae/vqvae_final_1250.pth",
        region_tokenizer_path: str = "models/appendages",
        seq_len: int = 900,
        use_vqvae: bool = True,
        duration_prediction: bool = False,
    ) -> None:
        """

        """
        super().__init__()  # initialize super class Dataset (from torch)

        self.region_tokenizer = RegionTokenizer(region_tokenizer_path)

        # --- vqvae ---
        self.vqvae = VQVAE(input_dim=12, codebook_size=512, embedding_dim=1024)
        vqvae_state_dict = torch.load(vqvae_path, map_location="cuda")
        self.vqvae.load_state_dict(vqvae_state_dict["model"])
        self.vqvae.to("cuda")
        self.vqvae.eval()
        self.use_vqvae = use_vqvae
        self.duration_prediction = duration_prediction

        # --- appendages + regions ---

        # temporary import for train data
        self.train_data = np.load(f"{data_path}/test_eeg.npy")
        self.val_data = np.load(f"{data_path}/validation_dataset.npy")

        train_data_joints = JointData(self.train_data)

        self.app_data = appendages(train_data_joints)  # (T, 12)
        self.app_data = self.region_tokenizer.scaler.transform(self.app_data)

        self.region_tokens = self.region_tokenizer.encode(
            torch.tensor(self.app_data)
        )  # (T,)

        # precompute VQVAE tokens
        self.vqvae_tokens_all = None
        if self.use_vqvae:
            print("Pre-computing VQ-VAE tokens...")
            self.vqvae_tokens_all = []
            chunk_size = 2048  # process in chunks
            with torch.no_grad():
                for i in tqdm(range(0, len(self.app_data), chunk_size)):
                    chunk = self.app_data[i: i + chunk_size]
                    chunk_tensor = (
                        torch.tensor(chunk, dtype=torch.float32)
                        .to("cuda")
                        .unsqueeze(0)
                    )
                    tokens = self.vqvae(chunk_tensor, return_toks=True)
                    self.vqvae_tokens_all.append(
                        tokens.cpu().numpy().flatten())
            self.vqvae_tokens_all = np.concatenate(self.vqvae_tokens_all)

        # --- eeg ---
        self.desired_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7',
                                 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

        self.raw_eeg = mne.io.read_raw_edf(f"{data_path}/eeg_test.edf")
        self.picks = mne.pick_channels(
            self.raw_eeg.ch_names, self.desired_channels)  # need to only have data channels
        self.raw_eeg = self.raw_eeg.copy().pick(self.picks)
        self.eeg_data, times = self.raw_eeg[:, :]
        self.eeg_data = torch.tensor(self.eeg_data, dtype=torch.float32)

        # sampling frequency
        original_sample_rate_hz = self.raw_eeg.info["sfreq"]
        target_sample_rate_hz = 30

        # instantiate the Resample transform
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate_hz,
            new_freq=target_sample_rate_hz
        )

        # apply the transform to your waveform
        downsampled_data: torch.Tensor = resampler(self.eeg_data)

        # FIXME fix if wrong or doing something weirdly
        self.eeg_data = downsampled_data.T
        start = self.eeg_data.shape[0] - self.train_data.shape[-2]
        self.eeg_data = self.eeg_data[start:, :]

        # --- sequences ---
        # regions will be a list of region sequences, each with shape (64,)
        self.regions = []
        self.appendages = []
        self.vqvae_token_crops = []
        self.eegs = []  # FIXME lol eegs

        # start at 0, go up to len(self.regions), and step by seq_len (30 seconds of data)
        for i in range(0, len(self.region_tokens), seq_len):
            region = self.region_tokens[i: i + seq_len]  # shape: (seq_len,)
            appendage = self.app_data[i: i + seq_len]  # shape: (seq_len, 12)
            # shape: (seq_len, num_eeg_channels)
            eeg = self.eeg_data[i: i + seq_len, :]

            # all regions are length seq_len
            if len(region) == seq_len and len(appendage) == seq_len:  # redundant
                self.regions.append(region)
                self.appendages.append(appendage)
                self.eegs.append(eeg)
                if self.use_vqvae:
                    self.vqvae_token_crops.append(
                        self.vqvae_tokens_all[i: i + seq_len]
                    )

    def __len__(self) -> int:
        return len(self.regions)

    def __getitem__(self, index: int) -> int:
        """
        Called when we do dataset[idx]

        __ means a "dunder" function for double underscore function.
        These are typically special functions that are called with
        special syntax. For example, __init__ is called when we
        initialize an object:
            E.g. a = RegionDataset(data_file="data/data.npy")
            calls __init__ with the parameter
        When we want to index into our object, we do:
            b = dataset[0]
        which converts to:
            b = dataset.__getitem__(0)
        """

        # returning (1, T) of vqvae tokens to input into PositionLLM
        # returning (T, 12) of appendage values
        # TODO add eegs to this if applicable
        if self.use_vqvae:
            vqvae_tokens = self.vqvae_token_crops[index]

            if not self.duration_prediction:
                return vqvae_tokens, self.appendages[index], self.eegs[index]
            else:
                # vqvae_tokens, appendage_values, durations, mask
                # vqvae_tokens: (T,)
                reversed_vqvae_toks = vqvae_tokens[::-1]
                durations = [1]
                for i in range(1, len(reversed_vqvae_toks)):
                    # counting backwards
                    current_tok = reversed_vqvae_toks[i]
                    prev_tok = reversed_vqvae_toks[i - 1]

                    if prev_tok == current_tok:
                        durations.append(durations[-1] + 1)
                    else:
                        durations.append(1)

                durations = durations[::-1]

                masks = [1]
                for i in range(len(vqvae_tokens) - 1):
                    current_tok = vqvae_tokens[i]
                    next_tok = vqvae_tokens[i + 1]

                    if next_tok == current_tok:
                        masks.append(0)
                    else:
                        masks.append(1)

                return {
                    "tokens": vqvae_tokens,
                    "values": self.appendages[index],
                    "durations": durations,
                    "masks": masks,
                    "eegs": self.eegs[index],
                }

        else:
            return self.regions[index], self.appendages[index], self.eegs[index]

    @staticmethod
    def collate_fn(batch):
        """
        Collate function to be used with DataLoader to batch data.
        """

        if isinstance(batch[0], dict):
            region_tokens = torch.tensor(
                [item["tokens"] for item in batch]).to(torch.int64)  # (B, T)
            appendage_values = torch.tensor(
                # (B, T, 12)
                [item["values"] for item in batch]).to(torch.float32)
            durations = torch.tensor([item["durations"] for item in batch]).to(
                torch.float32)  # (B, T)
            masks = torch.tensor([item["masks"]
                                 # (B, T)
                                  for item in batch]).to(torch.float32)
            eegs = torch.tensor([item["eegs"] for item in batch]).to(
                torch.float32)  # (B, T, num_eeg_channels)

            return {
                "tokens": region_tokens,
                "values": appendage_values,
                "durations": durations,
                "masks": masks,
                "eegs": eegs
            }

        region_tokens = torch.tensor(
            [item[0] for item in batch]).to(torch.int64)  # (B, T)
        appendage_values = torch.tensor(
            [item[1] for item in batch]).to(torch.float32)  # (B, T, 12)
        eeg_values = torch.stack(
            # (B, T, num_eeg_channels)
            [item[2] for item in batch]).to(torch.float32)

        return region_tokens, appendage_values, eeg_values
