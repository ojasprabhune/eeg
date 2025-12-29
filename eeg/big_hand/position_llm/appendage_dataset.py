from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import appendages
from .tokenizer import RegionTokenizer
from eeg.data_collection import JointData
from eeg.big_hand.position_llm.vqvae import VQVAE


class AppendageDataset(Dataset):
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
        AppendageDataset loading batches for tokenized appendage vectors.
        """
        super().__init__()  # initialize super class Dataset (from torch)

        self.region_tokenizer = RegionTokenizer(region_tokenizer_path)

        # import
        self.vqvae = VQVAE(input_dim=12, codebook_size=512, embedding_dim=1024)
        vqvae_state_dict = torch.load(vqvae_path, map_location="cuda")
        self.vqvae.load_state_dict(vqvae_state_dict["model"])
        self.vqvae.to("cuda")
        self.vqvae.eval()
        self.use_vqvae = use_vqvae

        self.train_data = np.load(f"{data_path}/training_dataset.npy")
        self.val_data = np.load(f"{data_path}/validation_dataset.npy")

        train_data_joints = JointData(self.train_data)

        self.app_data = appendages(train_data_joints)  # (T, 12)

        self.region_tokens = self.region_tokenizer.encode(
            torch.tensor(self.app_data)
        )  # (T,)

        self.app_data = self.region_tokenizer.scaler.transform(self.app_data)

        self.duration_prediction = duration_prediction

        # Precompute VQVAE tokens
        self.vqvae_tokens_all = None
        if self.use_vqvae:
            print("Pre-computing VQ-VAE tokens...")
            self.vqvae_tokens_all = []
            chunk_size = 2048  # Process in chunks
            with torch.no_grad():
                for i in tqdm(range(0, len(self.app_data), chunk_size)):
                    chunk = self.app_data[i : i + chunk_size]
                    chunk_tensor = (
                        torch.tensor(chunk, dtype=torch.float32)
                        .to("cuda")
                        .unsqueeze(0)
                    )
                    tokens = self.vqvae(chunk_tensor, return_toks=True)
                    self.vqvae_tokens_all.append(tokens.cpu().numpy().flatten())
            self.vqvae_tokens_all = np.concatenate(self.vqvae_tokens_all)

        # regions will be a list of region sequences, each with shape (64,)
        self.regions = []
        self.appendages = []
        self.vqvae_token_crops = []

        # start at 0, go up to len(self.regions), and step by seq_len (30 seconds of data)
        for i in range(0, len(self.region_tokens), seq_len):
            region = self.region_tokens[i : i + seq_len]  # shape: (seq_len,)
            appendage = self.app_data[i : i + seq_len]  # shape: (seq_len, 12)

            # all regions are length seq_len
            if len(region) == seq_len and len(appendage) == seq_len:  # redundant
                self.regions.append(region)
                self.appendages.append(appendage)
                if self.use_vqvae:
                    self.vqvae_token_crops.append(
                        self.vqvae_tokens_all[i : i + seq_len]
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
        if self.use_vqvae:
            vqvae_tokens = self.vqvae_token_crops[index]

            if not self.duration_prediction:
                return vqvae_tokens, self.appendages[index]
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
                    "masks": masks
                }

        else:
            return self.regions[index], self.appendages[index]

    def collate_fn(batch):
        """
        Collate function to be used with DataLoader to batch data.
        """

        if isinstance(batch[0], dict):
            region_tokens = torch.tensor([item["tokens"] for item in batch]).to(torch.int64)  # (B, T)
            appendage_values = torch.tensor([item["values"] for item in batch]).to(torch.float32)  # (B, T, 12)
            durations = torch.tensor([item["durations"] for item in batch]).to(torch.float32)  # (B, T)
            masks = torch.tensor([item["masks"] for item in batch]).to(torch.float32)  # (B, T)

            return {
                "tokens": region_tokens,
                "values": appendage_values,
                "durations": durations,
                "masks": masks
            }

        region_tokens = torch.tensor([item[0] for item in batch]).to(torch.int64)  # (B, T)
        appendage_values = torch.tensor([item[1] for item in batch]).to(torch.float32)  # (B, T, 12)

        return region_tokens, appendage_values