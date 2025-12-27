import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import appendages
from .tokenizer import RegionTokenizer
from eeg.data_collection import JointData


class AppendageDataset(Dataset):
    def __init__(self, data_path: str = "data/dataset", region_tokenizer_path: str = "models/appendages", seq_len: int = 900) -> None:
        """
        AppendageDataset loading batches for tokenized appendage vectors.
        """
        super().__init__()  # initialize super class Dataset (from torch)

        self.region_tokenizer = RegionTokenizer(region_tokenizer_path)

        self.train_data = np.load(f"{data_path}/training_dataset.npy")
        self.val_data = np.load(f"{data_path}/validation_dataset.npy")

        train_data_joints = JointData(self.train_data)

        self.app_data = appendages(train_data_joints)  # (T, 12)

        self.region_tokens = self.region_tokenizer.encode(
            torch.tensor(self.app_data)
        )  # (T,)

        self.app_data = self.region_tokenizer.scaler.transform(self.app_data)

        # regions will be a list of region sequences, each with shape (64,)
        self.regions = []
        self.appendages = []

        # start at 0, go up to len(self.regions), and step by seq_len (30 seconds of data)
        for i in range(0, len(self.region_tokens), seq_len):
            region = self.region_tokens[i: i + seq_len]  # shape: (seq_len,)
            appendage = self.app_data[i: i + seq_len]  # shape: (seq_len,)

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
        return self.regions[index], self.appendages[index]
