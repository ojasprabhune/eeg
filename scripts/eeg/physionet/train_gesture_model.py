"""
For the Physionet EEG Motor Movement/Imagery Dataset.

Trains GestureModel (transformer encoder + decoder-query architecture) to
predict a gesture class from one EEG epoch (raw channels, bandpower, CSP, or
DWT features - see input_type below).

train(input_type, fold) trains one model on one k-fold split and returns its
val accuracy. To run every fold of one input_type, or to sweep several input
types, just call train() again with different arguments - see
run_gesture_sweep.py for that loop.
"""

import math

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from eeg.gesture2hand import GestureModel
from eeg.gesture2hand.datasets.physio_net_gesture_dataset import get_cached_dataset

with open("config/gesture_model.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    experiment = config["experiment"]
    input_type = config["input_type"]
    k = config["k"]

    num_layers = config["num_layers"]
    decoder_num_layers = config["decoder_num_layers"]
    num_heads = config["num_heads"]
    embedding_dim = config["embedding_dim"]
    ffn_hidden_dim = config["ffn_hidden_dim"]
    encoder_dropout = config["encoder_dropout"]
    decoder_dropout = config["decoder_dropout"]

    device = config["device"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

num_channels = 64
num_features = 6
num_features_by_input_type = {
    "raw": num_channels,
    "bandpower": num_channels * num_features,
    "csp": num_features,
    "dwt": num_channels * num_features,
}


def select_input(
    raw: torch.Tensor,
    bp: torch.Tensor,
    csp: torch.Tensor,
    dwt: torch.Tensor,
    input_type: str,
) -> torch.Tensor:
    if input_type == "raw":
        return raw
    if input_type == "bandpower":
        return bp
    # csp/dwt are one flat feature vector per trial (B, C), so add time dim
    if input_type == "csp":
        return csp.unsqueeze(1)  # (B, num_features) -> (B, 1, num_features)
    if input_type == "dwt":
        return dwt.unsqueeze(
            1
        )  # (B, num_channels * num_features) -> (B, 1, num_channels * num_features)
    raise ValueError(f"unknown input_type: {input_type}")


def compute_f1(
    all_preds: torch.Tensor, all_labels: torch.Tensor, num_classes: int
) -> float:
    """Macro F1: average the per-class F1 score across every class."""
    f1s = []
    for cls in range(num_classes):
        tp = ((all_preds == cls) & (all_labels == cls)).sum().item()
        fp = ((all_preds == cls) & (all_labels != cls)).sum().item()
        fn = ((all_preds != cls) & (all_labels == cls)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    input_type: str,
    num_classes: int,
) -> tuple[float, float, float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)

    with torch.no_grad():
        for raw, bp, csp, dwt, labels in val_loader:
            features = select_input(raw, bp, csp, dwt, input_type).to(device)
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
                confusion_matrix[labels[i]][preds[i]] += 1

    model.train()
    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = compute_f1(torch.cat(all_preds), torch.cat(all_labels), num_classes)
    return avg_loss, accuracy, f1, confusion_matrix


def train(input_type: str, fold: int, print_confusion_matrix: bool) -> float:
    """
    Trains one fresh GestureModel on this input_type/fold combination
    (fold's examples held out as val, the other k-1 folds used for train).
    Returns the final val accuracy.
    """

    run_name = f"gesture_model_{experiment}_{input_type}_fold{fold}"

    print("\n=======================================")
    print(f"STARTING TRAINING FOR RUN: {run_name} FOR {epochs} EPOCHS")
    print("=======================================")

    # --- data ---

    dataset = get_cached_dataset(
        recordings_path="/Users/ojasprabhune/Documents/research/NORA/recordings/physio_net",
        num_recordings=100,
        k=k,
        fold=fold,
    )
    train_dataset = dataset.get_split("train")
    val_dataset = dataset.get_split("val")

    sample_weights, _ = train_dataset.get_sampler_weights()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,  # important for oversampling minority classes
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- model ---

    num_features = num_features_by_input_type[input_type]

    model = GestureModel(
        num_features=num_features,
        num_classes=train_dataset.num_classes,
        num_layers=num_layers,
        decoder_num_layers=decoder_num_layers,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {param_count:,}")

    # --- optimizer ---

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    def warmup_cosine_lr(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        total_steps = epochs * len(train_loader)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)

    if use_ckpt_path is not None:
        checkpoint = torch.load(use_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded model from checkpoint: {use_ckpt_path}")

    # --- train ---

    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": base_lr,
            "architecture": "GestureModel",
            "dataset": "gesture_dataset",
            "experiment": experiment,
            "input_type": input_type,
            "epochs": epochs,
            "k": k,
            "fold": fold,
        },
    )

    wandb.log({"param_count": param_count})
    model.train()

    val_acc = 0.0

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"{run_name} epoch {i + 1}")

        epoch_train_loss = 0.0
        n_batches = 0

        for raw, bp, csp, dwt, labels in train_loader:
            features = select_input(raw, bp, csp, dwt, input_type).to(
                device
            )  # (B, T, C)
            labels = labels.to(device)

            label_logits = model(features)  # out: (B, num_classes)
            loss = loss_fn(label_logits, labels)
            epoch_train_loss += loss.item()
            n_batches += 1

            optimizer.zero_grad()  # optimizer has access to all model params, grads -> 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly
            scheduler.step()  # steps lr

        val_loss, val_acc, val_f1, confusion_matrix = validate(
            model, val_loader, loss_fn, input_type, train_dataset.num_classes
        )
        run.log(
            {
                "train_loss": epoch_train_loss / n_batches,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "epoch": i + 1,
            }
        )
        epoch_tqdm.set_postfix(
            {"val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.3f}"}
        )

        if (i + 1) % 100 == 0 and print_confusion_matrix:
            print(f"\n{run_name} epoch {i + 1} validation confusion matrix:")
            print(confusion_matrix)

        if (i + 1) % save_every == 0 and save_ckpt_path is not None:
            latest_ckpt = {
                "epochs": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(latest_ckpt, f"{save_ckpt_path}_{run_name}_epoch_{i + 1}.pth")

    run.finish()

    if save_ckpt_path is not None:
        latest_ckpt = {
            "epochs": epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(latest_ckpt, f"{save_ckpt_path}_{run_name}.pth")

    print(f"TRAINING COMPLETE FOR RUN: {run_name}, final val_acc {val_acc:.3f}\n")

    return val_acc


if __name__ == "__main__":
    train(input_type=input_type, fold=0, print_confusion_matrix=False)
