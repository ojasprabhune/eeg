import numpy as np
import glob
import torch
from torch.utils.data import Dataset

from .utils import appendages
from .tokenizer import RegionTokenizer
from eeg.data_collection import JointData


class AppendageDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 900) -> None:
        """
        AppendageDataset loading batches for tokenized appendage vector.
        """
        super().__init__()  # initialize super class Dataset (from torch)

        self.region_tokenizer = RegionTokenizer("models/appendages")

        self.total_train_data: list = []
        self.total_val_data: list = []

        for npy_file in glob.glob(f"{data_path}/*.npy"):
            # take each npy file and put into list
            data = np.load(npy_file)
            split_idx = int(data.shape[1] * 0.8)  # index at 80% on time dim
            train_data = data[:, :split_idx, :]  # selecting 80%
            val_data = data[:, split_idx:, :]  # selecting 20%
            self.total_train_data.append(train_data)  # (# of npy files)
            self.total_val_data.append(val_data)  # (# of npy files)

        # concatenate along time dimension (down)
        self.train_data = np.concatenate(self.total_train_data, axis=1)
        self.val_data = np.concatenate(self.total_val_data, axis=1)

        """
        - test shapes:
        print("Train dataset shape:", self.train_data.shape)  # (2, T * 0.8, 63)
        print("Validation dataset shape:",
              self.val_data.shape)  # (2, T * 0.2, 63)

        """

        train_data_joints = JointData(self.train_data)

        self.app_data = appendages(train_data_joints)  # (T, 12)

        self.region_tokens = self.region_tokenizer.encode(
            torch.tensor(self.app_data)
        )  # (T,)

        # regions will be a list of region sequences, each with shape (64,)
        self.regions = []
        self.appendages = []

        # start at 0, go up to len(self.regions), and step by seq_len (30 seconds of data)
        for i in range(0, len(self.region_tokens), seq_len):
            region = self.region_tokens[i : i + seq_len]  # shape: (seq_len,)
            appendage = self.app_data[i : i + seq_len]  # shape: (seq_len,)

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
