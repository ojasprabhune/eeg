"""
SVM and LDA ran directly on the CSP and DWT per-trial feature vectors that
GestureDatasetmakes.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eeg.gesture2hand import GestureDataset

experiment = "6_letters"
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
    dataset = GestureDataset(
        experiment=experiment, mode="train", k=k, fold=fold, verbose=False
    )

    acc = evaluate(
        dataset.csp_epochs,
        dataset.labels,
        dataset.train_idx,
        dataset.val_idx,
        SVC(kernel="rbf"),
    )
    svm_accuracies_csp.append(acc)

    acc = evaluate(
        dataset.dwt_epochs,
        dataset.labels,
        dataset.train_idx,
        dataset.val_idx,
        SVC(kernel="rbf"),
    )
    svm_accuracies_dwt.append(acc)

    acc = evaluate(
        dataset.csp_epochs,
        dataset.labels,
        dataset.train_idx,
        dataset.val_idx,
        LinearDiscriminantAnalysis(),
    )
    lda_accuracies_csp.append(acc)

    acc = evaluate(
        dataset.dwt_epochs,
        dataset.labels,
        dataset.train_idx,
        dataset.val_idx,
        LinearDiscriminantAnalysis(),
    )
    lda_accuracies_dwt.append(acc)

    print(f"fold {fold} done")


def report(name: str, accuracies: list[float]) -> None:
    accuracies_arr = np.array(accuracies)
    print(f"{name}: {accuracies_arr.mean():.3f} +/- {accuracies_arr.std():.3f}")


print(f"\nchance level (1/{6}): {1 / 6:.3f}\n")
report("SVM + CSP", svm_accuracies_csp)
report("SVM + DWT", svm_accuracies_dwt)
report("LDA + CSP", lda_accuracies_csp)
report("LDA + DWT", lda_accuracies_dwt)
