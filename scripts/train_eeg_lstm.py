"""
Train an BLSTM to predict to Ojas's hand movements from his brain signals.
Implement all previous techniques like VQVAE tokenization on appendage vector
components.
"""

import yaml
import torch
import wandb
import numpy as np
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.eeg_data import LSTMDataset, EEGLSTM

with open("config/eeg_lstm.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    num_channels = config["num_channels"]
    embedding_dim = config["embedding_dim"]
    kernel_size_temporal = config["kernel_size_temporal"]
    dropout = config["dropout"]

    device = config["device"]
    batch_size = config["batch_size"]
    sequence_length = config["sequence_length"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

eeg_dataset: LSTMDataset = LSTMDataset(seq_len=sequence_length, device=device)
hand_dataloader = DataLoader(eeg_dataset, batch_size=32, shuffle=True)

model = EEGLSTM(
    vocab_size=vocab_size,
    num_layers=num_layers,
    num_channels=num_channels,
    embedding_dim=embedding_dim,
    kernel_size_temporal=kernel_size_temporal,
    dropout=dropout
)

optimizer = AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)
class_loss_fn = CrossEntropyLoss()

model.to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

def train():
    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": base_lr,
            "architecture": "BLSTM",
            "dataset": "eeg_dataset",
            "epochs": epochs,
        },
    )

    model.to(device)

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")


        iter_tqdm = tqdm(hand_dataloader, dynamic_ncols=True)
        for eeg, apps, tokens, durations, masks in iter_tqdm:
            # chunk: (B, T, C)

            eeg = eeg.to(device).float()
            apps = apps.to(device).float()
            tokens = tokens.to(device)

            token_logits = model(eeg) # out: (B, T, vocab_size), (B, T)
            token_logits = token_logits.transpose(1, 2) # (B, T, vocab_size) -> (B, vocab_size, T)

            loss = class_loss_fn(token_logits, tokens)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly


        if (i + 1) % save_every == 0:
            latest_ckpt = {
                "epochs": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(latest_ckpt, f"{save_ckpt_path}_epoch_{i + 1}.pth")

    run.finish()


train()

latest_ckpt = {
    "epochs": epochs,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

torch.save(latest_ckpt, save_ckpt_path)
