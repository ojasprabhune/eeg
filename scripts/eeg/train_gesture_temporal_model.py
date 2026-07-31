"""
Script to train the gesture temporal model on the gesture2hand dataset. The
model is a transformer encoder + attention-pooling architecture that takes in
one EEG epoch (one letter's motor execution window, either bandpower or raw
channels) and predicts the corresponding gesture class for that epoch.
"""

import math

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from eeg.gesture2hand import GestureDataset, GestureTemporalModel

# --- configuration -----------------------------------------------------------

with open("config/gesture_temporal_model.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    experiment = config["experiment"]
    input_type = config["input_type"]
    val_ratio = config["val_ratio"]

    d_model = config["d_model"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]

    device = config["device"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

num_features = 84 if input_type == "bandpower" else 14

# --- dataset & loss function ---

train_dataset = GestureDataset(
    experiment=experiment,
    mode="train",
    val_ratio=val_ratio,
    verbose=True,
)

val_dataset = GestureDataset(
    experiment=experiment,
    mode="val",
    val_ratio=val_ratio,
    verbose=False,
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


def select_input(raw: torch.Tensor, bp: torch.Tensor) -> torch.Tensor:
    return bp if input_type == "bandpower" else raw


# --- model ---

model = GestureTemporalModel(
    num_features=num_features,
    num_classes=train_dataset.num_classes,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
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


def validate() -> tuple[float, float, float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    val_confusion_matrix = torch.zeros(
        train_dataset.num_classes, train_dataset.num_classes, dtype=torch.int32
    )

    with torch.no_grad():
        for raw, bp, labels in val_loader:
            features = select_input(raw, bp).to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            for i in range(len(labels)):
                val_confusion_matrix[labels[i]][preds[i]] += 1

    model.train()
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    f1 = compute_f1(
        torch.cat(all_preds), torch.cat(all_labels), train_dataset.num_classes
    )
    return avg_loss, accuracy, f1, val_confusion_matrix


def train():
    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": base_lr,
            "architecture": "GestureTemporalModel",
            "dataset": "gesture_dataset",
            "experiment": experiment,
            "input_type": input_type,
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
        for raw, bp, labels in iter_tqdm:
            features = select_input(raw, bp).to(device)  # (B, T, C)
            labels = labels.to(device)

            label_logits = model(features)  # out: (B, num_classes)

            loss = loss_fn(label_logits, labels)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, grads -> 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()  # steps lr

        # --- end-of-epoch validation ---
        val_loss, val_acc, val_f1, val_confusion_matrix = validate()
        run.log(
            {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "epoch": i + 1}
        )
        epoch_tqdm.set_postfix(
            {
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.3f}",
                "val_f1": f"{val_f1:.3f}",
            }
        )

        if (i + 1) % 100 == 0:  # print every some epochs
            print(f"\nEpoch {i + 1} validation confusion matrix:")
            print(val_confusion_matrix)

        if (i + 1) % save_every == 0 and save_ckpt_path is not None:
            latest_ckpt = {
                "epochs": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(latest_ckpt, f"{save_ckpt_path}_epoch_{i + 1}.pth")

    run.finish()

    latest_ckpt = {
        "epochs": epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if save_ckpt_path is not None:
        torch.save(latest_ckpt, save_ckpt_path)


train()
