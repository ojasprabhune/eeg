import json
from pathlib import Path

import numpy as np

from eeg.gesture2hand import PhysioNetGestureDataset, load_physionet_data

skip_preprocessing = True

# CSP is refit on every load anyway, so don't bother fitting it here
data = load_physionet_data(
    load_from_saved=False, skip_preprocessing=skip_preprocessing, fit_csp=False
)
dataset = PhysioNetGestureDataset(data)

save_path = Path("~/Documents/research/NORA/recordings/physio_net/dataset").expanduser()
save_path.mkdir(parents=True, exist_ok=True)

np.save(save_path / "raw_epochs.npy", dataset.raw_epochs)
np.save(save_path / "bp_epochs.npy", dataset.bp_epochs)
np.save(save_path / "dwt_epochs.npy", dataset.dwt_epochs)
np.save(save_path / "labels.npy", dataset.labels)
np.save(save_path / "subject_ids.npy", dataset.subject_ids)

# how this dataset was made, checked by load_physionet_data when loading
meta = {
    "skip_preprocessing": skip_preprocessing,
    "motor_exec_sec": data["motor_exec_sec"],
    "epoch_start_sec": data["epoch_start_sec"],
    "num_epochs": len(dataset.labels),
    "num_subjects": len(np.unique(dataset.subject_ids)),
    "label_counts": np.bincount(dataset.labels).tolist(),
}
with open(save_path / "meta.json", "w") as meta_file:
    json.dump(meta, meta_file, indent=2)

print("Saved dataset settings:", meta)
