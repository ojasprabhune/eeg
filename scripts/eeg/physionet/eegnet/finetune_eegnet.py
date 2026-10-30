"""
Finetunes EEGNet using subject-specific transfer learning (SS-TL, Wang et al.
2020) on the PhysioNet EEG Motor Movement/Imagery Dataset.

For every held-out subject of every fold, starts from that fold's
cross-subject checkpoint (made by train_eegnet.py) and compares three scores
on the same test trials:
1. no calibration: the checkpoint as-is
2. batch norm refresh: batch norm statistics recomputed on the subject's
   calibration trials (no labels used)
3. fine-tuned: the checkpoint trained a few more epochs on the subject's
   calibration trials

Calibration trials are the first part of each class in recording order, and
test trials are the rest, so we always calibrate on the past and test on the
future.
"""

import numpy as np
import torch
import yaml
from torch import nn

from eeg.gesture2hand import EEGNet, load_physionet_data

with open("config/eegnet.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    experiment = config["experiment"]
    input_type = config["input_type"]
    skip_preprocessing = config["skip_preprocessing"]

    k = config["k"]
    device = config["device"]
    batch_size = config["batch_size"]

    save_ckpt_path = config["save_ckpt_path"]

# fine-tuning recipe from Wang et al. 2020: Adam, lr 1e-3, 5 epochs
finetune_epochs = 5
finetune_lr = 1e-3
calib_fraction = 1 / 3  # first third of each class calibrates, the rest is test

torch.manual_seed(42)


def load_model(fold: int) -> EEGNet:
    """Builds EEGNet and loads the cross-subject checkpoint of this fold."""
    model = EEGNet(
        vocab_size=config["vocab_size"],
        num_channels=config["num_channels"],
        num_samples=config["num_samples"],
        dropout=config["dropout"],
        kern_length=config["kern_length"],
        f1=config["f1"],
        f2=config["f2"],
        d=config["d"],
    ).to(device)

    # one forward pass so LazyLinear creates its weights before loading them
    dummy_batch = torch.zeros(1, 1, config["num_channels"], config["num_samples"])
    with torch.no_grad():
        model(dummy_batch.to(device))

    # same name train_eegnet.py saves the final checkpoint under
    run_name = f"eegnet_{experiment}_{input_type}_fold{fold}"
    checkpoint = torch.load(f"{save_ckpt_path}_{run_name}.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])

    model.eval()
    return model


def apply_max_norm(model: EEGNet) -> None:
    """Max-norm constraints from the paper, same as train_eegnet.py."""
    with torch.no_grad():
        model.depthwise_conv2d.weight.data = torch.renorm(
            model.depthwise_conv2d.weight.data.flatten(1),
            p=2,
            dim=0,
            maxnorm=1.0,
        ).view_as(model.depthwise_conv2d.weight)
        model.vocab_projection.weight.data = torch.renorm(
            model.vocab_projection.weight.data, p=2, dim=0, maxnorm=0.25
        )


def predict(model: EEGNet, x: torch.Tensor) -> torch.Tensor:
    """Returns the predicted class of every trial in x, shape (N,)."""
    model.eval()
    with torch.no_grad():
        logits = model(x)  # (N, num_classes)
    return logits.argmax(dim=1)


def refresh_batch_norm(model: EEGNet, x: torch.Tensor) -> None:
    """
    Batch norm keeps a running mean/variance of what it saw in training.
    Recompute them from this subject's trials only. No labels are used.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None  # plain average instead of a running average
            module.train()  # only batch norm updates its stats, dropout stays off

    with torch.no_grad():
        model(x)

    model.eval()


def finetune(model: EEGNet, x: torch.Tensor, y: torch.Tensor) -> None:
    """
    Trains all layers a few more epochs on this subject's calibration trials.

    The shape if x is (N, 1, C, T) and y is (N,), where N is the number of calibration
    trials, C is the number of EEG channels, and T is the number of time samples.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    model.train()

    for _ in range(finetune_epochs):
        perm = torch.randperm(len(y))  # new random order every epoch

        for start in range(0, len(y), batch_size):
            batch = perm[start : start + batch_size]

            logits = model(x[batch])  # (B, num_classes)
            loss = loss_fn(logits, y[batch])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            apply_max_norm(model)

    model.eval()


def add_to_confusion(
    confusion: np.ndarray, y_true: torch.Tensor, y_pred: torch.Tensor
) -> None:
    """Rows = true class, cols = predicted class."""
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        confusion[true][pred] += 1


if __name__ == "__main__":
    method_names = ["no calibration", "batch norm refresh", "fine-tuned (SS-TL)"]
    accs = {name: [] for name in method_names}  # one accuracy per subject
    confusions = {name: np.zeros((4, 4), dtype=np.int64) for name in method_names}
    n_calib_per_subject = []
    class_names = []  # filled in when the first fold loads

    for fold in range(k):
        # same k, fold and seed as train_eegnet.py, so these val subjects were
        # never seen by this fold's checkpoint
        data = load_physionet_data(
            split="subject",
            k=k,
            fold=fold,
            load_from_saved=True,
            skip_preprocessing=skip_preprocessing,
            fit_csp=False,
        )
        raw_epochs = data["raw_epochs"]
        labels = data["labels"]
        subject_ids = data["subject_ids"]
        val_idx = data["val_idx"]
        class_names = data["class_names"]

        for subject in np.unique(subject_ids[val_idx]):
            # val_idx is sorted, so each subject's trials stay in recording order
            subject_idx = val_idx[subject_ids[val_idx] == subject]

            # split each class by time: first part calibrates, rest is test
            calib_idx, test_idx = [], []
            for cls in range(data["num_classes"]):
                cls_idx = subject_idx[labels[subject_idx] == cls]
                n_calib = round(len(cls_idx) * calib_fraction)
                calib_idx.extend(cls_idx[:n_calib])
                test_idx.extend(cls_idx[n_calib:])

            # (N, T, C) -> (N, 1, C, T), same as the training script
            x_calib = torch.tensor(raw_epochs[calib_idx]).transpose(1, 2).unsqueeze(1)
            x_test = torch.tensor(raw_epochs[test_idx]).transpose(1, 2).unsqueeze(1)
            y_calib = torch.tensor(labels[calib_idx])
            y_test = torch.tensor(labels[test_idx])

            x_calib, x_test = x_calib.to(device), x_test.to(device)
            y_calib, y_test = y_calib.to(device), y_test.to(device)

            # reload the checkpoint for every method so nothing carries over
            # from the previous method or the previous subject
            model = load_model(fold)
            preds_none = predict(model, x_test)

            model = load_model(fold)
            refresh_batch_norm(model, x_calib)
            preds_bn = predict(model, x_test)

            model = load_model(fold)
            finetune(model, x_calib, y_calib)
            preds_ft = predict(model, x_test)

            for name, preds in zip(method_names, [preds_none, preds_bn, preds_ft]):
                accs[name].append((preds == y_test).float().mean().item())
                add_to_confusion(confusions[name], y_test.cpu(), preds.cpu())

            n_calib_per_subject.append(len(calib_idx))
            print(
                f"fold {fold} {subject}: "
                f"none {accs[method_names[0]][-1]:.3f}, "
                f"bn {accs[method_names[1]][-1]:.3f}, "
                f"ft {accs[method_names[2]][-1]:.3f} "
                f"({len(calib_idx)} calib, {len(test_idx)} test)"
            )

    # --- results -------------------------------------------------------------

    print(
        f"\n{len(n_calib_per_subject)} subjects, "
        f"{np.mean(n_calib_per_subject):.1f} calibration trials per subject on average"
    )
    print(f"classes: {class_names}\n")

    for name in method_names:
        print(
            f"{name}: {np.mean(accs[name]):.3f} +- {np.std(accs[name]):.3f} (mean +- std over subjects)"
        )
        print(confusions[name], "\n")
