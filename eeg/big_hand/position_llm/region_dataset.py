import numpy as np
import glob
import os

import torch
from torch.utils.data import Dataset

from .utils import process_deltas
from .tokenizer import DeltaTokenizer, RegionTokenizer


class RegionDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 900) -> None:
        """
        RegionDataset loading batches for tokenized regions tokenized on delta tokens.
        """

        super().__init__()  # initialize super class Dataset (from torch)

        self.delta_tokenizer = DeltaTokenizer()
        self.region_tokenizer = RegionTokenizer("models/delta_tokens")

        self.original_data: np.ndarray = np.empty(
            (0, 63)
        )  # initialize with 0 rows and 63 channels

        os.chdir(data_path)
        for npy_file in glob.glob("*.npy"):
            # append data from each npy file in data path downwards (by row)
            self.original_data = np.append(
                self.original_data, np.load(npy_file), axis=0
            )  # (T, 63)

        self.deltas: np.ndarray = process_deltas(self.original_data)  # (T, 63)
        self.delta_tokens: torch.Tensor = self.delta_tokenizer.encode(
            self.deltas
        )  # (T, 63)
        self.region_tokens: torch.Tensor = self.region_tokenizer.encode(
            self.delta_tokens
        )  # (T,)

        # regions will be a list of region sequences, each with shape (64,)
        self.regions = []
        self.deltas = []

        # start at 0, go up to len(self.regions), and step by seq_len (30 seconds of data)
        for i in range(0, len(self.region_tokens), seq_len):
            region = self.region_tokens[i : i + seq_len]  # shape: (seq_len,)
            # all regions are length seq_len
            if len(region) == seq_len:
                self.regions.append(region)

        # start at 0, go up to len(self.deltas), and step by seq_len (30 seconds of data)
        for i in range(0, len(self.delta_tokens), seq_len):
            delta = self.delta_tokens[i : i + seq_len]  # shape: (seq_len,)
            # all deltas are length seq_len
            if len(delta) == seq_len:
                self.deltas.append(delta)

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, index: int) -> int:
        """
        Called when we do dataset[idx]

        __ means a "dunder" function for double underscore function.
        These are typically special functions that are called with
        special syntax. For example, __init__ is called when we
        initialize an object:
            E.g. a = DeltaDataset(data_file="data/data.npy")
            calls __init__ with the parameter
        When we want to index into our object, we do:
            b = dataset[0]
        which converts to:
            b = dataset.__getitem__(0)
        """
        return self.regions[index], self.deltas[index]
