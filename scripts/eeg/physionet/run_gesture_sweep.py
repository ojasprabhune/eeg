"""
For the Physionet EEG Motor Movement/Imagery Dataset.

Runs every (model, input_type, fold) combination for the two gesture
classifiers, one at a time in this one process for a specific
experiment defined in the configuration.

This must be run multiple times with different experiment configurations.
"""

from train_gesture_model import train as train_gesture_model
from train_gesture_temporal_model import train as train_gesture_temporal_model

input_types = ["bandpower", "csp", "dwt"]
k = 1

results = {}

for input_type in input_types:
    val_acc = train_gesture_model(
        input_type=input_type,
        print_confusion_matrix=False,
    )
    results[f"gesture_model/{input_type}"] = val_acc

for input_type in input_types:
    val_acc = train_gesture_temporal_model(
        input_type=input_type,
        print_confusion_matrix=False,
    )
    results[f"gesture_temporal_model/{input_type}"] = val_acc

print("\nsweep done, final val_acc per run:")
for name, val_acc in results.items():
    print(f"{name}: {val_acc:.3f}")
