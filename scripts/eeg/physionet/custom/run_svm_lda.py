"""
For the Physionet EEG Motor Movement/Imagery Dataset.

SVM and LDA ran directly on the CSP and DWT per-trial feature vectors that
load_physionet_data makes.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eeg.gesture2hand import load_physionet_data

skip_preprocessing = True  # must match the saved dataset (see meta.json)
k = 5


def evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    classifier,
) -> float:
    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_val = features[val_idx]
    y_val = labels[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_val, y_val)
    return accuracy


svm_accuracies_csp = []
svm_accuracies_dwt = []
lda_accuracies_csp = []
lda_accuracies_dwt = []

for fold in range(k):
    data = load_physionet_data(
        split="subject", k=k, fold=fold, skip_preprocessing=skip_preprocessing
    )

    acc = evaluate(
        data["csp_epochs"],
        data["labels"],
        data["train_idx"],
        data["val_idx"],
        SVC(kernel="rbf"),
    )
    svm_accuracies_csp.append(acc)

    acc = evaluate(
        data["dwt_epochs"],
        data["labels"],
        data["train_idx"],
        data["val_idx"],
        SVC(kernel="rbf"),
    )
    svm_accuracies_dwt.append(acc)

    acc = evaluate(
        data["csp_epochs"],
        data["labels"],
        data["train_idx"],
        data["val_idx"],
        LinearDiscriminantAnalysis(),
    )
    lda_accuracies_csp.append(acc)

    acc = evaluate(
        data["dwt_epochs"],
        data["labels"],
        data["train_idx"],
        data["val_idx"],
        LinearDiscriminantAnalysis(),
    )
    lda_accuracies_dwt.append(acc)

    print(f"fold {fold} done")


def report(name: str, accuracies: list[float]) -> None:
    accuracies_arr = np.array(accuracies)
    print(f"{name}: {accuracies_arr.mean():.3f} +/- {accuracies_arr.std():.3f}")


print(f"\nchance level (1/4): {1 / 4:.3f}\n")
report("SVM + CSP", svm_accuracies_csp)
report("SVM + DWT", svm_accuracies_dwt)
report("LDA + CSP", lda_accuracies_csp)
report("LDA + DWT", lda_accuracies_dwt)
