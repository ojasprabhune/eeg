import torch
import torchaudio
import mne
from torch.utils.data import Dataset

mne.set_log_level("WARNING") # to suppress info messages

class HandDataset(Dataset):
    def __init__(
        self,
        num_folders: int,
        data_path: str = "/var/log/thavamount/eeg_dataset/motor_eeg/1.0.0",
        seq_len: int = 800,
    ) -> None:
        """

        """
        super().__init__()  # initialize super class Dataset (from torch)

        self.raws = []
        for i in range(num_folders):
            for j in [3, 4, 7, 8]:
                self.raws.append(mne.io.read_raw_edf(
                    f"{data_path}/S00{i + 1}/S00{i + 1}R0{j}.edf", preload=True))
            for j in [11, 12]:
                self.raws.append(mne.io.read_raw_edf(
                    f"{data_path}/S00{i + 1}/S00{i + 1}R{j}.edf", preload=True))

        self.raw: mne.io.Raw = mne.concatenate_raws(self.raws, preload=True)

        # band-pass filter
        self.filtered = self.raw.copy().filter(l_freq=0.1, h_freq=75)

        # notch filtering
        self.filtered = self.filtered.notch_filter(freqs=60)

        # sampling frequency
        orig_sr = 160
        new_sr = 200

        self.resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
        self.resampled = self.resampler(torch.tensor(self.filtered.get_data()).float())

        # --- chunking ---
        self.chunks = []

        # start at 0, go up to len(self.resampled[-1]), and step by seq_len
        for i in range(0, len(self.resampled[-1]), seq_len):
            chunk = self.resampled[:, i : i + seq_len] # C, seq_len

            # all chunks are length seq_len except maybe last
            if len(chunk[-1]) == seq_len:
                self.chunks.append(chunk)


    def __len__(self) -> int:
        return len(self.chunks)


    def __getitem__(self, index: int):
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

        return self.chunks[index]
