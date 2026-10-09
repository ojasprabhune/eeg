from pathlib import Path

import numpy as np

from eeg.gesture2hand import PhysioNetGestureDataset, load_physionet_data

data = load_physionet_data(load_from_saved=False)
dataset = PhysioNetGestureDataset(data)

save_path = Path("~/Documents/research/NORA/recordings/physio_net/dataset").expanduser()
save_path.mkdir(parents=True, exist_ok=True)

np.save(save_path / "raw_epochs1.npy", dataset.raw_epochs)
np.save(save_path / "bp_epochs1.npy", dataset.bp_epochs)
np.save(save_path / "dwt_epochs1.npy", dataset.dwt_epochs)
np.save(save_path / "csp_epochs1.npy", dataset.csp_epochs)
np.save(save_path / "labels1.npy", dataset.labels)
np.save(save_path / "subject_ids1.npy", dataset.subject_ids)
