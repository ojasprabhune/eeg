"""
For the Physionet EEG Motor Movement/Imagery Dataset.

run_svm_lda.py has SVC and LDA that are solved in one shot without epochs,
so there's no overfitting. SGDClassifier with loss="hinge" is sklearn's own
linear SVM trained via stochastic gradient descent. It updates its weights a
little at a time across many passes over the data.

Trains one SGD linear SVM per input type on the subject split, logging
train/val accuracy to wandb every epoch, and stops early once val accuracy
hasn't beaten its best in `patience` epochs - once it plateaus, then stop.
"""

import os

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import wandb
from eeg.gesture2hand import PhysioNetGestureDataset

os.environ["WANDB_SILENT"] = "True"  # make wandb shh
np.seterr(all="ignore")

input_types = ["csp", "dwt"]
max_epochs = 3000
patience = 300  # stop once val_acc hasn't beaten its best in this many epochs


def run_one(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    run_name: str,
) -> tuple[int, float]:
    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_val = features[val_idx]
    y_val = labels[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # every class label that can show up, needed up front because
    # partial_fit only ever sees one batch at a time and can't infer on its
    # own that a class missing from an early batch still exists
    classes = np.unique(labels)

    classifier = SGDClassifier(loss="hinge", random_state=42)

    rng = np.random.RandomState(42)

    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={"max_epochs": max_epochs, "patience": patience},
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_y_preds = np.zeros_like(y_val)
    epochs_since_best = 0

    for epoch in range(1, max_epochs + 1):
        # shuffle the training rows into a new random order every epoch so
        # model can't memorize order
        perm = rng.permutation(len(X_train))

        # partial means one pass, not full fit so we can stop it
        classifier.partial_fit(X_train[perm], y_train[perm], classes=classes)
        y_preds = classifier.predict(X_val)

        train_acc = classifier.score(X_train, y_train)
        val_acc = classifier.score(X_val, y_val)
        run.log({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_y_preds = y_preds  # predictions at the best epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epochs_since_best >= patience:
            break

    run.log({"best_epoch": best_epoch, "best_val_acc": best_val_acc})
    run.finish()

    # --- diagnostics at the best epoch ---------------------------------------
    # classes: 0 left fist, 1 right fist, 2 both fists, 3 both feet
    # y // 2 gives the block: 0 = left/right fist, 1 = both fists/both feet
    print(f"\n{run_name} confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_val, best_y_preds, labels=[0, 1, 2, 3]))
    print("balanced acc:", balanced_accuracy_score(y_val, best_y_preds))

    same_block = (best_y_preds // 2) == (y_val // 2)
    print("block acc:", same_block.mean())
    print(
        "within-block acc:",
        (best_y_preds[same_block] == y_val[same_block]).mean(),
    )

    return best_epoch, best_val_acc


dataset = PhysioNetGestureDataset(split="subject", load_from_saved=True)

for input_type in input_types:
    features = dataset.csp_epochs if input_type == "csp" else dataset.dwt_epochs

    best_epoch, best_val_acc = run_one(
        features,
        dataset.labels,
        dataset.train_idx,
        dataset.val_idx,
        run_name=f"sgd_svm_{input_type}",
    )

    print(f"{input_type}: best epoch {best_epoch}, best val_acc {best_val_acc:.3f}\n")
