import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from eeg.position_llm.data import DeltaData

training_data = DeltaData("/home/prabhune/projects/research/2026/eeg/data/test.npy")
<<<<<<< HEAD
# test_data = DeltaData("test")

train_dataloader = DataLoader(training_data, batch_size=501, shuffle=False)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
=======

train_dataloader = DataLoader(training_data, batch_size=501, shuffle=False)
>>>>>>> eeg_data

train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")

# print(train_features[:, 3:6]) # expected relative xyz values
# print(torch.diff(train_features[:, 3:6], dim=0))  # expected delta xyz values

train_features_deltas = torch.diff(train_features, dim=0)

max_delta = torch.max(train_features_deltas)
min_delta = torch.min(train_features_deltas)
print("max", max_delta.item())
print("min", min_delta.item())


train_features_deltas = torch.trunc(train_features_deltas * 10) / 10

print(train_features_deltas[:, 24:27])

plt.plot(train_features_deltas[:, 24:27].numpy())
