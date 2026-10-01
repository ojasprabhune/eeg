"""
Runs every (model, input_type, fold) combination for the two gesture
classifiers, one at a time in this one process.
"""

from train_gesture_model import train as train_gesture_model
from train_gesture_temporal_model import train as train_gesture_temporal_model

input_types = ["bandpower", "csp", "dwt"]
k = 1

results = {}

for input_type in input_types:
    for fold in range(k):
        val_acc = train_gesture_model(input_type=input_type, fold=fold)
        results[f"gesture_model/{input_type}/fold{fold + 1}"] = val_acc

for input_type in input_types:
    for fold in range(k):
        val_acc = train_gesture_temporal_model(input_type=input_type, fold=fold)
        results[f"gesture_temporal_model/{input_type}/fold{fold + 1}"] = val_acc

print("\nsweep done, final val_acc per run:")
for name, val_acc in results.items():
    print(f"{name}: {val_acc:.3f}")
