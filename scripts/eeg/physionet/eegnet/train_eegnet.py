"""
Trains EEGNet (see model class file) with series of convolutions in order to
classify a class out of 4 classes from EEG data.
"""

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from eeg.gesture2hand import EEGNet, PhysioNetGestureDataset, load_physionet_data

with open("config/eegnet.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    experiment = config["experiment"]
    input_type = config["input_type"]

    num_epochs = config["num_epochs"]
    device = config["device"]
    batch_size = config["batch_size"]
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

            features = features.transpose(1, 2).unsqueeze(1)  # (B, 1, C, T)

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


def train(input_type: str, print_confusion_matrix: bool) -> float:
    """
    Trains one fresh GestureModel on this input_type/fold combination
    (fold's examples held out as val, the other k-1 folds used for train).
    Returns the final val accuracy.
    """

    run_name = f"eegnet_{experiment}_{input_type}"

    print("\n=======================================")
    print(f"STARTING TRAINING FOR RUN: {run_name} FOR {epochs} EPOCHS")
    print("=======================================")

    # --- data ---
    data = load_physionet_data(
        split="subject",
        num_epochs=num_epochs,
        load_from_saved=True,
        verbose=True,
    )
    train_dataset = PhysioNetGestureDataset(data, mode="train")
    val_dataset = PhysioNetGestureDataset(data, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # --- model ---

    num_features = num_features_by_input_type[input_type]

    model = (
        EEGNet(
            vocab_size=config["vocab_size"],
            num_channels=config["num_channels"],
            num_samples=config["num_samples"],
            dropout=config["dropout"],
            kern_length=config["kern_length"],
            f1=config["f1"],
            f2=config["f2"],
            d=config["d"],
        )
        .to(device)
        .eval()
    )

    raw, bp, csp, dwt, labels = train_dataset[0]
    features = select_input(raw, bp, csp, dwt, input_type).to(device)  #  (T, C)

    dummy_batch = features.transpose(0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, C, T)
    with torch.no_grad():
        dummy_out = model(dummy_batch)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {param_count:,}")

    # --- optimizer ---

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

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
            "architecture": "EEGNet",
            "dataset": "physionet",
            "experiment": experiment,
            "input_type": input_type,
            "epochs": epochs,
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

            features = features.transpose(1, 2).unsqueeze(1)  # (B, 1, C, T)

            label_logits = model(features)  # out: (B, num_classes)
            loss = loss_fn(label_logits, labels)
            epoch_train_loss += loss.item()
            n_batches += 1

            optimizer.zero_grad()  # optimizer has access to all model params, grads -> 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly

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
    train(input_type=input_type, print_confusion_matrix=False)
