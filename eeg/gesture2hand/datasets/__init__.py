from .utils import Colors

try:  # temporal_dataset needs mne, absent in minimal containers
    from .temporal_dataset import TemporalDataset
except Exception:
    TemporalDataset = None

try:  # gesture_dataset needs mne + pyxdf, absent in minimal containers
    from .gesture_dataset import GestureDataset
except Exception:
    GestureDataset = None
