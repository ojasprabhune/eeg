"""
Script to train the temporal model on the gesture2hand dataset. The model is a
transformer-based architecture that takes in sequences of bandpower features and
predicts the corresponding hand gesture labels.
"""

import math

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

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

# --- dataset & loss function ---

train_dataset = TemporalDataset(
    mode="train",
    seq_len=sequence_length,
    stride=stride,
    device=device,
    verbose=True,
    data_mode="bp",
)

sample_weights, class_weights = train_dataset.get_sampler_weights()

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,  # important for oversampling minority classes
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

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

# --- optimizer ---


def warmup_cosine_lr(step: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    total_steps = epochs * len(train_loader)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)

if use_ckpt_path is not None:
    checkpoint = torch.load(use_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.last_epoch = checkpoint["epochs"] * len(train_loader)
    print(f"Loaded model from checkpoint: {use_ckpt_path}")


def train():
    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": base_lr,
            "architecture": "TransformerEncoder",
            "dataset": "temporal_dataset",
            "epochs": epochs,
        },
    )

    wandb.log({"param_count": param_count})
    model.to(device)
    model.train()

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")

        iter_tqdm = tqdm(train_loader, dynamic_ncols=True)
        for eeg, bp, apps, tokens, labels, durations, masks in iter_tqdm:
            # chunk: (B, T, C)

            bp = bp.to(device)
            labels = labels.to(device)

            label_logits = model(bp)  # out: (B, vocab_size)

            loss = loss_fn(label_logits, labels)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()  # steps lr

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
