"""
Train a linear regression model to map EEG data to appendage positions.
"""

import yaml
import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.eeg_data import HandDataset, LabramModel

with open("config/labram_basic_lin.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    base_lr = float(config["hyperparameters"]["base_lr"])
    epochs = config["hyperparameters"]["epochs"]
    run_name = config["hyperparameters"]["run_name"]

hand_dataset: HandDataset = HandDataset(num_folders=5, new_sfreq=200, label_sfreq=50)

hand_dataloader = DataLoader(hand_dataset,
                            batch_size=32,
                            shuffle=True)

device = "cuda"

model = LabramModel(embedding_dim=512)
optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)

loss_fn = CrossEntropyLoss()

model.to(device)


def train():
    # start a new wandb run to track this script.
    run = wandb.init(
        name=run_name,
        # set the wandb entity where your project will be logged (generally your team name).
        entity="prabhuneojas-evergreen-valley-high-school",
        # set the wandb project where this run will be logged.
        project="eeg",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": base_lr,
            "architecture": "linear",
            "dataset": "region_dataset",
            "epochs": epochs,
        },
    )

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")

        iter_tqdm = tqdm(hand_dataloader, dynamic_ncols=True)
        for chunk, label_chunk in iter_tqdm:
            # chunk: (B, num_channels, T)
            hand_pos_logits = model(chunk.to(device).float()) # out: (B, T, vocab_size)

            hand_pos_logits = hand_pos_logits.transpose(1, 2)
            
            loss = loss_fn(hand_pos_logits, label_chunk.to(device))

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly

    run.finish()


train()
