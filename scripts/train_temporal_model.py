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
    train_fraction = config.get("train_fraction", 1.0)

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

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,  # important for oversampling minority classes
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

# --- validation ---

def compute_f1(all_preds: torch.Tensor, all_labels: torch.Tensor, nc: int) -> float:
    """Compute macro F1 score."""
    f1s = []
    for cls in range(nc):
        tp = ((all_preds == cls) & (all_labels == cls)).sum().item()
        fp = ((all_preds == cls) & (all_labels != cls)).sum().item()
        fn = ((all_preds != cls) & (all_labels == cls)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if len(f1s) > 0 else 0.0

def validate() -> tuple[float, float, float]:
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
            logits = model(bp)
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
    f1 = compute_f1(torch.cat(all_preds), torch.cat(all_labels), vocab_size)
    return avg_loss, accuracy, f1


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

            label_logits = model(bp)  # out: (B, vocab_size)

            loss = loss_fn(label_logits, labels)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()  # steps lr

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
