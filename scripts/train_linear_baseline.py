"""
Script to train the linear baseline model on the gesture2hand dataset. The model
is a mean-pooling MLP that takes in sequences of bandpower features and predicts
the corresponding hand gesture labels.

Supports both 4-class (Fist, Left, Fingers, Open) and 3-class (excluding Open)
classification, controlled by the `num_classes` and `exclude_open` fields in the
config file.

Usage:
    uv run scripts/train_linear_baseline.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from eeg.gesture2hand import EEGLinearBaseline, TemporalDataset

with open("config/linear_temporal.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    stride = config["stride"]
    val_ratio = config["val_ratio"]
    num_features = config["num_features"]
    num_classes = config["num_classes"]
    exclude_open = config["exclude_open"]
    train_fraction = config.get("train_fraction", 1.0)
    dropout = config["dropout"]

    device = config["device"]
    batch_size = config["batch_size"]
    sequence_length = config["seq_length"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]
    weight_decay = config["weight_decay"]

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
    train_fraction=train_fraction,
)

val_dataset = TemporalDataset(
    mode="val",
    seq_len=sequence_length,
    stride=stride,
    device=device,
    verbose=True,
    data_mode="bp",
)

sample_weights, class_weights = train_dataset.get_sampler_weights()

if exclude_open:
    # --- filter out chunks whose majority label is Open (class 3) ----------
    # and remap labels: 0=Fist, 1=Left, 2=Fingers (drop 3=Open)

    def filter_and_remap(dataset):
        """Remove chunks with majority-label Open and remap 0-2."""
        chunk_labels = np.array(
            [np.bincount(c, minlength=4).argmax() for c in dataset.label_chunks_split]
        )
        keep = chunk_labels != 3  # drop Open

        dataset.eeg_chunks_split = dataset.eeg_chunks_split[keep]
        dataset.bp_chunks_split = dataset.bp_chunks_split[keep]
        dataset.app_chunks_split = dataset.app_chunks_split[keep]
        dataset.token_chunks_split = dataset.token_chunks_split[keep]
        dataset.label_chunks_split = dataset.label_chunks_split[keep]

        # remap: labels 0, 1, 2 stay the same; label 3 was removed
        # (no actual remapping needed since we only kept 0-2)

    filter_and_remap(train_dataset)
    filter_and_remap(val_dataset)

    # recompute sampler weights for filtered train set
    sample_weights, class_weights = train_dataset.get_sampler_weights()

    # trim class weights to 3 classes (drop the Open weight at index 3)
    class_weights = class_weights[:3]

    print(f"Filtered to {num_classes} classes (excluding Open)")
    print(f"  Train chunks: {len(train_dataset)}")
    print(f"  Val chunks:   {len(val_dataset)}")

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,  # important for oversampling minority classes
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

# --- model ---

model = EEGLinearBaseline(
    num_features=num_features,
    num_classes=num_classes,
    dropout=dropout,
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


optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)

if use_ckpt_path is not None:
    checkpoint = torch.load(use_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.last_epoch = checkpoint["epochs"] * len(train_loader)
    print(f"Loaded model from checkpoint: {use_ckpt_path}")


# --- validation ---


def compute_f1(all_preds: torch.Tensor, all_labels: torch.Tensor, nc: int) -> float:
    """Compute macro F1 score (no sklearn needed)."""
    f1s = []
    for cls in range(nc):
        tp = ((all_preds == cls) & (all_labels == cls)).sum().item()
        fp = ((all_preds == cls) & (all_labels != cls)).sum().item()
        fn = ((all_preds != cls) & (all_labels == cls)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def validate() -> tuple[float, float, float]:
    """Run one pass over the validation set, return (loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for eeg, bp, apps, tokens, labels, durations, masks in val_loader:
            bp = bp.to(device)
            labels = labels.to(device)

            logits = model(bp)  # (B, num_classes)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * bp.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    model.train()
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    f1 = compute_f1(torch.cat(all_preds), torch.cat(all_labels), num_classes)
    return avg_loss, accuracy, f1


# --- training ---


def train():
    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": base_lr,
            "architecture": "EEGLinearBaseline",
            "dataset": "temporal_dataset",
            "epochs": epochs,
            "num_classes": num_classes,
            "exclude_open": exclude_open,
            "train_fraction": train_fraction,
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

            label_logits = model(bp)  # out: (B, num_classes)

            loss = loss_fn(label_logits, labels)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # --- end-of-epoch validation ---
        val_loss, val_acc, val_f1 = validate()
        run.log({"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "epoch": i + 1})
        epoch_tqdm.set_postfix({"val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.3f}", "val_f1": f"{val_f1:.3f}"})

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
