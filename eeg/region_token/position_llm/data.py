import numpy as np

import torch
from torch.utils.data import Dataset

from eeg.region_token.position_llm import RegionTokenizer


class RegionDataset(Dataset):
    def __init__(self, data_file: str) -> None:
        """
        RegionDataset loading batches for tokenized deltas.
        """
        super().__init__()  # initialize super class Dataset (from torch)
        # load original raw position values from npy file
        self.original_data: np.ndarray = np.load(data_file)  # (T, 63)
        # find difference downward
        self.original_data = np.diff(self.original_data, axis=0)  # (T, 63)

        print(
            "Delta minimum:",
            self.original_data.min(),
            "\nDelta maximum:",
            self.original_data.max(),
        )

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
