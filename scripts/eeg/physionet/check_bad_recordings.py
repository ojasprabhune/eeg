from collections import defaultdict
from pathlib import Path

import mne

root = Path("~/Documents/research/NORA/recordings/physio_net").expanduser()
by_key = defaultdict(list)
for p in sorted(root.rglob("*.edf")):
    r = mne.io.read_raw_edf(p, preload=False, verbose=False)
    by_key[(r.info["sfreq"], len(r.ch_names))].append(p.stem)

for key, stems in by_key.items():
    print(key, len(stems), stems[:5])
