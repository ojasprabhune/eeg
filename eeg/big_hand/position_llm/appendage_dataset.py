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

        # regions will be a list of region sequences, each with shape (64,)
        self.regions = []
        self.appendages = []

        # start at 0, go up to len(self.regions), and step by seq_len (30 seconds of data)
        for i in range(0, len(self.region_tokens), seq_len):
            region = self.region_tokens[i: i + seq_len]  # shape: (seq_len,)
            appendage = self.app_data[i: i + seq_len]  # shape: (seq_len, 12)

            # all regions are length seq_len
            if len(region) == seq_len and len(appendage) == seq_len:  # redundant
                self.regions.append(region)
                self.appendages.append(appendage)
        

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
            vqvae_tokens = self.vqvae(torch.tensor(self.appendages[index]).unsqueeze(0).to("cuda").to(torch.float32), return_toks=True)
            vqvae_tokens = vqvae_tokens.squeeze(0).cpu().numpy()

            if not self.duration_prediction:
                with torch.no_grad():
                    return vqvae_tokens, self.appendages[index]
            else:
                # vqvae_tokens, appendage_values, durations, mask
                # vqvae_tokens: (T,)
                
                # Vectorized duration calculation
                # Find where tokens change (going backwards)
                reversed_vqvae_toks = vqvae_tokens[::-1]
                token_changes = np.concatenate([[True], reversed_vqvae_toks[1:] != reversed_vqvae_toks[:-1]])
                
                # Calculate run lengths
                run_lengths = np.zeros(len(reversed_vqvae_toks), dtype=np.float32)
                current_length = 1
                for i in range(len(reversed_vqvae_toks)):
                    if token_changes[i]:
                        current_length = 1
                    run_lengths[i] = current_length
                    current_length += 1
                
                # Apply log10 transform and reverse back
                durations = np.log10(run_lengths)
                durations = durations[::-1]
                
                # Vectorized mask calculation
                # mask is 1 when next token is different, 0 when same
                masks = np.concatenate([
                    vqvae_tokens[:-1] != vqvae_tokens[1:],  # 1 where tokens differ
                    [True]  # last position always gets mask=1
                ]).astype(np.float32)
                
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
            region_tokens = torch.tensor(np.array([item["tokens"] for item in batch])).to(torch.int64)  # (B, T)
            appendage_values = torch.tensor(np.array([item["values"] for item in batch])).to(torch.float32)  # (B, T, 12)
            durations = torch.tensor(np.array([item["durations"] for item in batch])).to(torch.float32)  # (B, T)
            masks = torch.tensor(np.array([item["masks"] for item in batch])).to(torch.float32)  # (B, T)

            return {
                "tokens": region_tokens,
                "values": appendage_values,
                "durations": durations,
                "masks": masks
            }

        region_tokens = torch.tensor([item[0] for item in batch]).to(torch.int64)  # (B, T)
        appendage_values = torch.tensor([item[1] for item in batch]).to(torch.float32)  # (B, T, 12)

        return region_tokens, appendage_values