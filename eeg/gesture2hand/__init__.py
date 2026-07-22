from .datasets.utils import Colors
from .utils import gesture_experiments, get_gesture_class

try:  # heavy optional deps (braindecode/torch models) absent in containers
    from .datasets import TemporalDataset
    from .models import EEGLinearBaseline, TemporalModel
except Exception:
    TemporalDataset = EEGLinearBaseline = TemporalModel = None
