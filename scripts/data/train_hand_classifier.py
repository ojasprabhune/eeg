import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path


class HandClassifier(nn.Module):
    def __init__(self, vocab_size: int = 4, embedding_dim: int = 128):

        super().__init__()

        self.fc1 = nn.Linear(63, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.fc4 = nn.Linear(embedding_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.head(x)

        return x


class HandClassifierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str = "~/Documents/research/recordings/videos/trial_5/part_3/part_3_hands.npy",
        labels_path: str = "~/Documents/research/recordings/videos/trial_5/part_3/part_3_labels.npy",
        seq_len: int = 900,
    ) -> None:

        full_data = []
        full_data = torch.tensor(np.load(Path(data_path).expanduser()))  # (2, T, 63)
        self.data = full_data[1, :, :]  # (T, 63)

        self.labels = torch.tensor(np.load(Path(labels_path).expanduser()))  # (T,)

        # trim both to the same length
        self.labels = self.labels[self.labels.shape[0] - self.data.shape[0] :]
        print(self.data.shape)
        print(self.labels.shape)

        self.labels = self.labels - 1  # convert from 1-4 to 0-3

        self.data_chunks = []
        self.label_chunks = []

        for i in range(0, len(self.data), seq_len):
            data_chunk = self.data[i : i + seq_len]  # shape: (seq_len,)
            label_chunk = self.labels[i : i + seq_len]  # shape: (seq_len, 12)

            # all regions are length seq_len
            if len(data_chunk) == seq_len and len(label_chunk) == seq_len:
                self.data_chunks.append(data_chunk)
                self.label_chunks.append(label_chunk)

    def __len__(self) -> int:
        return len(self.data_chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data_chunks[idx], self.label_chunks[idx]


hand_dataset = HandClassifierDataset()
model = HandClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 5000

hand_dataloader = DataLoader(hand_dataset, batch_size=32, shuffle=True)

save_ckpt_path = "~/Documents/research/hand_classifier/hand_classifier.pth"


def train():
    run = wandb.init(
        name="hand_classifier",
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": 5e-5,
            "architecture": "NN",
            "dataset": "hand_classifier",
            "epochs": epochs,
        },
    )

    model.to(device)

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")

        iter_tqdm = tqdm(hand_dataloader, dynamic_ncols=True)
        for data_chunk, label_chunk in iter_tqdm:
            # chunk: (B, T, C)

            data_chunk = data_chunk.to(device).to(torch.float32)
            label_chunk = label_chunk.to(device).to(torch.int64)

            label_logits = model(data_chunk)  # out: (B, T, vocab_size)

            # (B, T, vocab_size) -> (B, vocab_size, T)
            label_logits = label_logits.transpose(1, 2)

            loss = loss_fn(label_logits, label_chunk)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly

        if (i + 1) % 1000 == 0:
            latest_ckpt = {
                "epochs": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(
                latest_ckpt, Path(f"{save_ckpt_path}_epoch_{i + 1}.pth").expanduser()
            )

    run.finish()


train()

latest_ckpt = {
    "epochs": epochs,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

torch.save(latest_ckpt, Path(save_ckpt_path).expanduser())
