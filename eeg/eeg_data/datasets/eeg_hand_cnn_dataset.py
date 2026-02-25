import mne
import numpy as np

import torch
from torch.utils.data import Dataset

mne.set_log_level("WARNING")  # to suppress info messages


class HandDatasetCNN(Dataset):
    def __init__(
        self,
        num_folders: int,
        data_path: str = "/var/log/thavamount/eeg_dataset/motor_eeg/1.0.0",
    ) -> None:
        """
        Dataset for hand movement classification using EEG data. Loads and
        preprocesses the data.
        """
        super().__init__()  # initialize super class Dataset (from torch)

        # --- eeg data ---
        self.raws = []
        for i in range(num_folders):
            subject = f"S{i+1:03d}"  # zero-pads to 3 digits (001, 002)

            for j in [3, 4, 7, 8]:
                run = f"R{j:02d}"  # zero-pads to 2 digits (01, 02)
                self.raws.append(
                    mne.io.read_raw_edf(
                        f"{data_path}/{subject}/{subject}{run}.edf",
                        preload=True
                    )
                )

            for j in [11, 12]:
                run = f"R{j:02d}"
                self.raws.append(
                    mne.io.read_raw_edf(
                        f"{data_path}/{subject}/{subject}{run}.edf",
                        preload=True
                    )
                )

        self.raw: mne.io.Raw = mne.concatenate_raws(self.raws, preload=True)
        self.filtered = self.raw.copy().filter(
            l_freq=0.1, h_freq=75)  # band-pass filter
        self.filtered = self.filtered.notch_filter(freqs=60)  # notch filtering
        self.filtered.set_eeg_reference(
            "average", projection=False)  # common average reference
        self.filtered.filter(8, 30, method="iir", iir_params=dict(
            order=4, ftype="butter"))  # butterworth

        self.eeg_data = self.filtered.get_data()  # (C, T)

        # --- events and labels ---
        # events is shape (n_events, 3) where each row is (sample, 0, event_id)
        self.events, self.event_ids = mne.events_from_annotations(self.raw)

        # epochs is shape (n_epochs, C, T)
        self.epochs = mne.Epochs(self.raw, events=self.events)
        self.epochs = self.epochs.get_data()  # (n_epochs, C, T)

        # every row: (sample, event_id)
        self.events = np.delete(self.events, 1, axis=1)

        # --- chunking ---
        self.eeg_chunks = self.epochs  # (n_epochs, C, T)
        self.label_chunks = []
        self.mask = []

        # start at 0 and go up to total number of events
        for i in range(0, self.events.shape[0]):
            if i % 30 == 0:
                continue  # skip those that are too close to the start of the recording since we won't have a full chunk of data for those
            else:
                # event_id - 1 to convert from 1, 2, 3 to 0, 1, 2
                self.label_chunks.append(self.events[i, 1] - 1)

        for label in self.label_chunks:
            if label == 0:
                self.mask.append(1)
            else:
                self.mask.append(5)

        # eeg chunks shape: (n_epochs, C, T)
        # label chunks shape: (n_epochs,)
        # mask shape: (n_epochs,)

        # --- train-val split ---
        # index at 80% on time dim
        self.split_idx = int(self.eeg_chunks.shape[0] * 0.5)
        # selecting 80%
        self.train_eeg_chunks = self.eeg_chunks[:self.split_idx, :, :]
        # selecting 80%
        self.train_label_chunks = self.label_chunks[:self.split_idx]
        self.mask = self.mask[:self.split_idx]  # selecting 80%

        # selecting 20%
        self.val_eeg_chunks = self.eeg_chunks[self.split_idx:, :, :]
        # selecting 20%
        self.val_label_chunks = self.label_chunks[self.split_idx:]

    def __len__(self) -> int:
        """
        Return the number of training data chunks.
        """

        return len(self.train_label_chunks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get training data chunk and corresponding label chunk based on index.

        Parameters:
            index (int): Index of the data chunk to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The EEG data chunk and
            corresponding label chunk.

        """

        return self.train_eeg_chunks[index], torch.tensor(self.train_label_chunks[index]), torch.tensor(self.mask[index])

    def get_validation_data(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get validation data chunk and corresponding label chunk based on index.

        Parameters:
            index (int): Index of the data chunk to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The EEG data chunk and
            corresponding label chunk.

        """

        return self.val_eeg_chunks[index], torch.tensor(self.val_label_chunks[index]), torch.tensor(self.mask[index])
