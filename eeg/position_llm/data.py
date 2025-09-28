from .tokenizer import DeltaTokenizer
from ..data_collection.utils import normalize

import numpy as np

import torch
from torch.utils.data import Dataset


class DeltaDataset(Dataset):
    def __init__(self, data_file: str) -> None:
        """
        Only using 24 for index finger tip positions.
        """
        super().__init__()
        self.original_data: np.ndarray = np.load(data_file)[
            :, 24
        ]  # shape: (total_T, 63) => (total_T,)
        self.original_data = np.diff(self.original_data, axis=0)
        print(self.original_data.min(), self.original_data.max())
        self.data: np.ndarray = normalize(
            self.original_data,
            self.original_data.max(),
            self.original_data.min(),
            10,
            -10,
        )
        self.data = self.data.round(decimals=1)
        print(self.data.min(), self.data.max())

        # deltas will be a list of delta sequences, each with shape (64,)
        self.deltas = []

        # start at 0, go up to len(self.data), and step by 64
        for i in range(0, len(self.data), 64):
            delta = self.data[i : i + 64]  # shape: (64,)
            self.deltas.append(delta)

    def __len__(self):
        return len(self.deltas)

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
        delta: np.ndarray = self.deltas[index]  # shape: (64, 63)
        idx_finger_tip = torch.tensor(delta)  # x-pos of index finger tip

        return idx_finger_tip
