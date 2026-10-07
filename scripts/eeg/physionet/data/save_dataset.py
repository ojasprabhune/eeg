from pathlib import Path

import numpy as np

from eeg.gesture2hand import PhysioNetGestureDataset

dataset = PhysioNetGestureDataset(split="subject", load_from_saved=False, verbose=True)

save_path = Path("~/Documents/research/NORA/recordings/physio_net/dataset").expanduser()
save_path.mkdir(parents=True, exist_ok=True)

np.save(save_path / "raw_epochs.npy", dataset.raw_epochs)
np.save(save_path / "bp_epochs.npy", dataset.bp_epochs)
np.save(save_path / "dwt_epochs.npy", dataset.dwt_epochs)
np.save(save_path / "csp_epochs.npy", dataset.csp_epochs)
np.save(save_path / "labels.npy", dataset.labels)
np.save(save_path / "subject_ids.npy", dataset.subject_ids)
