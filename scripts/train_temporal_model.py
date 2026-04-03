"""
"""

import yaml
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, WeightedRandomSampler

from eeg.gesture2hand.datasets import TemporalDataset

with open("config/temporal.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    stride = config["stride"]
    val_ratio = config["val_ratio"]

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

# --- block-stratified random split ---

chunks_per_block = max(1, sequence_length // stride)
n_blocks = n_chunks // chunks_per_block

block_labels = np.array(
    [
        int(
            label_chunks[b * chunks_per_block : (b + 1) * chunks_per_block].mean() > 0.5
        )
        for b in range(n_blocks)
    ]
)

rng = np.random.RandomState(42)

open_blocks = np.where(block_labels == 1)[0]
closed_blocks = np.where(block_labels == 0)[0]
rng.shuffle(open_blocks)
rng.shuffle(closed_blocks)

n_val_open = max(1, int(len(open_blocks) * val_ratio))
n_val_closed = max(1, int(len(closed_blocks) * val_ratio))

val_block_ids = np.concatenate([open_blocks[:n_val_open], closed_blocks[:n_val_closed]])
train_block_ids = np.concatenate(
    [open_blocks[n_val_open:], closed_blocks[n_val_closed:]]
)

def blocks_to_chunks(block_ids: np.ndarray) -> np.ndarray:
    chunk_ids = []
    for b in block_ids:
        start = b * chunks_per_block
        end = min(start + chunks_per_block, n_chunks)
        chunk_ids.extend(range(start, end))
    return np.array(chunk_ids)


train_idx = blocks_to_chunks(train_block_ids)
val_idx = blocks_to_chunks(val_block_ids)

train_feats = all_bp_chunks[train_idx]
train_labels = label_chunks[train_idx]
val_feats = all_bp_chunks[val_idx]
val_labels = label_chunks[val_idx]

print(f"Train: {len(train_idx)} chunks ({train_labels.mean():.1%} open)")
print(f"Val:   {len(val_idx)} chunks ({val_labels.mean():.1%} open)")

# --- class balancing ---

n_open = (train_labels == 1).sum()
n_closed = (train_labels == 0).sum()
pos_weight = torch.tensor([n_closed / max(n_open, 1)], dtype=torch.float32).to(device)
print(
    f"Class balance — open: {n_open}, closed: {n_closed}, pos_weight: {pos_weight.item():.3f}"
)

sample_weights = np.where(train_labels == 1, n_closed / max(n_open, 1), 1.0)
sampler = WeightedRandomSampler(
    weights=sample_weights.tolist(),
    num_samples=len(train_labels),
    replacement=True,
)

train_ds = TensorDataset(
    torch.from_numpy(train_feats), torch.from_numpy(train_labels).float()
)
val_ds = TensorDataset(
    torch.from_numpy(val_feats), torch.from_numpy(val_labels).float()
)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, sampler=sampler, drop_last=True
)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


# --- model ---

model = EEGTemporalModel(
    num_features=num_features,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
).to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {model_type}, parameters: {param_count:,}")
wandb.log({"param_count": param_count})

# --- optimizer ---

optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)


def warmup_cosine_lr(step: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    total_steps = epochs * len(train_loader)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

