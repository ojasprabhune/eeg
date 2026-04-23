"""Evaluate temporal model on validation set. Prints accuracy, F1, and confusion matrix."""

import argparse

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.gesture2hand import TemporalDataset, TemporalModel

GESTURE_LABELS = ["Fist", "Left", "Fingers", "Open"]

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/temporal.yaml")
parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch (omit for final)")
parser.add_argument("--best", action="store_true", help="Load best checkpoint")
parser.add_argument("--exclude-open", action="store_true", help="Exclude Open class from metrics")
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = yaml.safe_load(config_file)

device = config["device"]
vocab_size = config["vocab_size"]

# --- dataset ---

val_dataset = TemporalDataset(
    eeg_data_path=config.get("eeg_data_path", "/var/log/thavamount/eeg_dataset/home_eeg"),
    hand_data_path=config.get("hand_data_path", "/var/log/thavamount/eeg_dataset/hand_data"),
    vqvae_path=config.get("vqvae_path", "/var/log/thavamount/eeg_ckpts/eeg_vqvae/vqvae_final_1250.pth"),
    region_tokenizer_path=config.get("region_tokenizer_path", "models/appendages"),
    mode="val",
    seq_len=config["seq_length"],
    stride=config["stride"],
    device=device,
    verbose=False,
)

val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# --- model ---

model = TemporalModel(
    num_features=config["num_features"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    dropout=config["dropout"],
    vocab_size=vocab_size,
).to(device)

save_ckpt_path = config["save_ckpt_path"]
if args.best:
    ckpt_path = f"{save_ckpt_path}_best.pth"
elif args.epoch is not None:
    ckpt_path = f"{save_ckpt_path}_epoch_{args.epoch}.pth"
else:
    ckpt_path = save_ckpt_path

state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict["model"])
print(f"Loaded: {ckpt_path}")

# --- inference ---

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for _, bp, _, _, labels, _, _ in tqdm(val_loader, desc="Evaluating"):
        bp = bp.to(device)
        logits = model(bp)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- metrics ---

def print_metrics(labels, preds, class_ids, title=""):
    names = [GESTURE_LABELS[i] for i in class_ids]
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
    print(f"\nAccuracy: {accuracy_score(labels, preds):.4f}")
    print(f"Samples: {len(preds)}\n")
    print(classification_report(
        labels, preds,
        target_names=names,
        labels=class_ids,
        digits=4,
        zero_division=0,
    ))
    cm = confusion_matrix(labels, preds, labels=class_ids)
    print("Confusion Matrix (rows=true, cols=pred):")
    header = "          " + "".join(f"{g:>10}" for g in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{names[i]:>10}" + "".join(f"{v:>10}" for v in row))

print_metrics(all_labels, all_preds, list(range(vocab_size)), "All Classes")

if args.exclude_open:
    mask = all_labels != 3
    print_metrics(all_labels[mask], all_preds[mask], [0, 1, 2], "Excluding Open (true label)")
