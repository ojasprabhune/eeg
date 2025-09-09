from torch.utils.data import DataLoader

from eeg.position_llm.data import DeltaData

training_data = DeltaData(
    "/home/prabhune/projects/research/2026/eeg/data/open_fist_front.npy"
)
# test_data = DeltaData("test")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(train_features[:, 3:6])
