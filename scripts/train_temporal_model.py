"""
Script to train the temporal model on the gesture2hand dataset. The model is a
transformer-based architecture that takes in sequences of bandpower features and
predicts the corresponding hand gesture labels.
"""

import math

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

import wandb
from eeg.gesture2hand import TemporalDataset, TemporalModel

with open("config/temporal.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    stride = config["stride"]
    val_ratio = config["val_ratio"]
    num_features = config["num_features"]
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    vocab_size = config["vocab_size"]

    device = config["device"]
    batch_size = config["batch_size"]
    sequence_length = config["seq_length"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

train_dataset = TemporalDataset(
    mode="train", seq_len=sequence_length, stride=stride, device=device, verbose=True
)
weights, class_weights = train_dataset.get_sampler_weights()

sampler = WeightedRandomSampler(
    weights=weights, num_samples=len(weights), replacement=True
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True
)

loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

# --- model ---

model = TemporalModel(
    num_features=num_features,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    vocab_size=vocab_size,
).to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model parameters: {param_count:,}")
# wandb.log({"param_count": param_count})

# --- optimizer ---

optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)


def warmup_cosine_lr(step: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    total_steps = epochs * len(train_loader)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)
