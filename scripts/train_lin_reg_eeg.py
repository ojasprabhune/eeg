"""
Train a linear regression model to map EEG data to appendage positions.
"""

import yaml
import torch
import wandb
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.eeg_data import EEGDataset
from eeg.eeg_data import EEGRegressionModel

with open("config/eeg_regression.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    base_lr = float(config["hyperparameters"]["base_lr"])
    epochs = config["hyperparameters"]["epochs"]
    run_name = config["hyperparameters"]["run_name"]

eeg_dataset: EEGDataset = EEGDataset()

eeg_dataloader = DataLoader(eeg_dataset,
                            batch_size=32,
                            shuffle=True,
                            collate_fn=EEGDataset.collate_fn)

device = "cuda"

model = EEGRegressionModel(14, 12)
optimizer = AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)

class_loss_fn = CrossEntropyLoss()

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

        iter_tqdm = tqdm(eeg_dataloader, dynamic_ncols=True)
        for batch in iter_tqdm:

            eegs = batch["eegs"]  # (B, T, num_eeg_channels)
            app_values = batch["values"]  # (B, T, 12)

            eegs = eegs.to(device)
            app_values = app_values.to(device)

            pred_app_values = model(eegs)  # (B, T, 12)

            loss = class_loss_fn(pred_app_values, app_values)

            iter_tqdm.set_postfix(loss)
            run.log(loss)

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly

    run.finish()


train()
