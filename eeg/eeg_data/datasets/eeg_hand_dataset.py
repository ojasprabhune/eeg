import mne
import numpy as np

import torch
from torch.utils.data import Dataset

mne.set_log_level("WARNING") # to suppress info messages

class HandDataset(Dataset):
    def __init__(
        self,
        num_folders: int,
        new_sfreq: int = 200,
        label_sfreq: int = 50,
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
        self.events, self.event_ids = mne.events_from_annotations(self.raw)

        # band-pass filter
        self.filtered = self.raw.copy().filter(l_freq=0.1, h_freq=75)

        # notch filtering
        self.filtered = self.filtered.notch_filter(freqs=60)

        # sampling frequency
        self.resampled, self.events = self.filtered.copy().resample(sfreq=new_sfreq, events=self.events)
        self.resampled_label, self.events_label = self.filtered.copy().resample(sfreq=label_sfreq, events=self.events)

        self.eeg_data = self.resampled.get_data()
        self.events_label = np.delete(self.events_label, 1, axis=1) # every row: (sample, event_id)

        self.labels = np.zeros(self.eeg_data.shape[-1], dtype=np.int64)

        for i in range(len(self.events_label)):
            if i == len(self.events_label) - 1:
                ts = self.events_label[i, 0]
                self.labels[ts:] = self.events_label[i, 1] - 1
            else:
                ts = self.events_label[i, 0]
                next_ts = self.events_label[i+1, 0]
                self.labels[ts:next_ts] = self.events_label[i, 1] - 1

        # prepend SOS token
        sos = torch.tensor([3])
        self.labels = torch.cat((sos, torch.tensor(self.labels)), dim=0).numpy()

        # mask
        self.mask = [1]
        for i in range(len(self.labels) - 1):
            current_label = self.labels[i]
            next_label = self.labels[i + 1]

            if next_label == current_label:
                self.mask.append(1e-9)
            else:
                self.mask.append(1)

        # --- chunking ---
        self.chunks = []
        self.label_chunks = []
        self.masks = []

        self.label_seq_len = seq_len // (new_sfreq // label_sfreq)

        # start at 0, go up to len(self.resampled[-1]), and step by seq_len
        for i in range(0, len(self.eeg_data[-1]), seq_len):
            chunk = self.eeg_data[:, i : i + seq_len] # C, seq_len

            # all chunks are length seq_len except maybe last
            if len(chunk[-1]) == seq_len:
                self.chunks.append(chunk)
        for i in range(0, self.labels.shape[-1], self.label_seq_len):
            label_chunk = self.labels[i : i + self.label_seq_len] # seq_len
            mask = self.mask[i : i + self.label_seq_len] # seq_len
            mask = torch.tensor(mask)

            # all chunks are length seq_len except maybe last
            if label_chunk.shape[-1] == self.label_seq_len and len(mask) == self.label_seq_len:
                self.label_chunks.append(label_chunk)
                self.masks.append(mask)


    def __len__(self) -> int:
        return len(self.chunks)


    def __getitem__(self, index: int):
        """
        Called when we do dataset[idx]
        """

        return self.chunks[index], self.label_chunks[index], self.masks[index]
