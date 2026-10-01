"""
For the Physionet EEG Motor Movement/Imagery Dataset.

run_svm_lda.py has SVC and and LDA that are solved in one shot without epochs,
so theres no overfitting.GDClassifier with loss="hinge" is sklearn's own. It is
a linear SVM trained via stochastic gradient descent. It updates its weights a
little at a time across many passes over the data.

Trains one SGD-linear-SVM per (input_type, fold), logging train/val
accuracy to wandb every epoch, and stops each one early once val accuracy
hasn't beaten its best in patience epochs - once it plateaus, then stop.
"""

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import wandb
from eeg.gesture2hand.datasets.physio_net_gesture_dataset import get_cached_dataset

experiment = "common_8_letters"
num_recordings = 5
k = 5
input_types = ["csp", "dwt"]
max_epochs = 3000
patience = 300  # stop once val_acc hasn't beaten its best in this many epochs


def run_fold(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
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
    epochs_since_best = 0

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    quit()

    for epoch in range(1, max_epochs + 1):
        # shuffle the training rows into a new random order every epoch so
        # model can't memorize order
        perm = rng.permutation(len(X_train))

        # partial means one pass, not full fit so we can stop it
        classifier.partial_fit(X_train[perm], y_train[perm], classes=classes)

        train_acc = classifier.score(X_train, y_train)
        val_acc = classifier.score(X_val, y_val)
        run.log({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epochs_since_best >= patience:
            break

    run.log({"best_epoch": best_epoch, "best_val_acc": best_val_acc})
    run.finish()

    return best_epoch, best_val_acc


for input_type in input_types:
    fold_results = []

    for fold in range(k):
        dataset = get_cached_dataset(
            recordings_path="/Users/ojasprabhune/Documents/research/NORA/recordings/physio_net",
            num_recordings=num_recordings,
            k=k,
            fold=fold,
        )

        features = dataset.csp_epochs if input_type == "csp" else dataset.dwt_epochs

        best_epoch, best_val_acc = run_fold(
            features,
            dataset.labels,
            dataset.train_idx,
            dataset.val_idx,
            dataset.num_classes,
            run_name=f"sgd_svm_{input_type}_fold{fold}",
        )
        fold_results.append((best_epoch, best_val_acc))
        print(
            f"{input_type} fold {fold}: stopped at epoch {best_epoch}, "
            f"best val_acc {best_val_acc:.3f}"
        )

    accs = np.array([acc for _, acc in fold_results])
    epochs_stopped = np.array([ep for ep, _ in fold_results])
    print(
        f"\n{input_type}: best val_acc {accs.mean():.3f} +/- {accs.std():.3f}, "
        f"stopped at epoch {epochs_stopped.mean():.0f} +/- {epochs_stopped.std():.0f}\n"
    )
