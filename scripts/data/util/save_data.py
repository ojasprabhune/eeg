import glob
import numpy as np

total_train_data: list = []
total_val_data: list = []

for npy_file in glob.glob("data/*.npy"):
    data = np.load(npy_file)
    split_idx = int(data.shape[1] * 0.8)  # index at 80% on time dim
    train_data = data[:, :split_idx, :]  # selecting 80%
    val_data = data[:, split_idx:, :]  # selecting 20%
    total_train_data.append(train_data)  # (# of npy files)
    total_val_data.append(val_data)  # (# of npy files)

# concatenate along time dimension (down)
train_data = np.concatenate(total_train_data, axis=1)
val_data = np.concatenate(total_val_data, axis=1)

print("Train dataset shape:", train_data.shape)  # (2, T * 0.8, 63)
print("Validation dataset shape:",
      val_data.shape)  # (2, T * 0.2, 63)

np.save("data/dataset/training_dataset.npy", train_data)
np.save("data/dataset/validation_dataset.npy", val_data)
