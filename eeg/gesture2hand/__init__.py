from .datasets.utils import Colors
from .utils import gesture_experiments, get_gesture_class

try:  # heavy optional deps (braindecode/torch models) absent in containers
    from .datasets import GestureDataset, TemporalDataset
    from .models import (
        EEGLinearBaseline,
        GestureModel,
        GestureTemporalModel,
        TemporalModel,
    )
except Exception:
    TemporalDataset = GestureDataset = EEGLinearBaseline = TemporalModel = None
    GestureModel = GestureTemporalModel = None
