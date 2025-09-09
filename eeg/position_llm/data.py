import numpy as np
from torch.utils.data import Dataset


class DeltaData(Dataset):
    def __init__(self, data_file: str, transform=None) -> None:
        super().__init__()
        self.data = np.load(data_file)
        self.transform = transform

    # retrieve number of samples in dataset
    def __len__(self):
        return len(self.data)  # returns # of time steps

    def __getitem__(self, index) -> int:
        deltas = self.data[index]
        if self.transform:
            deltas = self.transform(deltas)
        return deltas
