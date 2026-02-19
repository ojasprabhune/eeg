from curses import raw

import mne
import numpy as np

import torch
from torch.utils.data import Dataset

mne.set_log_level("WARNING") # to suppress info messages

class HandDatasetCNN(Dataset):
    def __init__(
        self,
        num_folders: int,
        data_path: str = "/var/log/thavamount/eeg_dataset/motor_eeg/1.0.0",
    ) -> None:
        """

        """
        super().__init__()  # initialize super class Dataset (from torch)

        # --- eeg data ---
        self.raws = []
        for i in range(num_folders):
            for j in [3, 4, 7, 8]:
                self.raws.append(mne.io.read_raw_edf(
                    f"{data_path}/S00{i + 1}/S00{i + 1}R0{j}.edf", preload=True))
            for j in [11, 12]:
                self.raws.append(mne.io.read_raw_edf(
                    f"{data_path}/S00{i + 1}/S00{i + 1}R{j}.edf", preload=True))

        self.raw: mne.io.Raw = mne.concatenate_raws(self.raws, preload=True)
        self.filtered = self.raw.copy().filter(l_freq=0.1, h_freq=75) # band-pass filter
        self.filtered = self.filtered.notch_filter(freqs=60) # notch filtering
        self.eeg_data = self.filtered.get_data() # (C, T)

        # --- events and labels ---
        self.events, self.event_ids = mne.events_from_annotations(self.raw) # events is shape (n_events, 3) where each row is (sample, 0, event_id)

        self.epochs = mne.Epochs(self.raw, events=self.events) # epochs is shape (n_epochs, C, T)
        self.epochs = self.epochs.get_data() # (n_epochs, C, T)

        self.events = np.delete(self.events, 1, axis=1) # every row: (sample, event_id)

        # --- chunking ---
        self.eeg_chunks = self.epochs # (n_epochs, C, T)
        self.label_chunks = []

        # start at 0 and go up to total number of events 
        for i in range(0, self.events.shape[0]):
            if i % 30 == 0:
                continue # skip those that are too close to the start of the recording since we won't have a full chunk of data for those
            else:
                self.label_chunks.append(self.events[i, 1] - 1) # event_id - 1 to convert from 1, 2, 3 to 0, 1, 2
        
        # eeg chunks shape: (n_epochs, C, T)
        # label chunks shape: (n_epochs,)

    def __len__(self) -> int:
        return len(self.label_chunks)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Called when we do dataset[idx]
        """

        return self.eeg_chunks[index], torch.tensor(self.label_chunks[index])
