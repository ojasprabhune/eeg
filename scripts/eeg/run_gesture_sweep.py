"""
Runs every (model, input_type, fold) combination for the two gesture
classifiers, one at a time in this one process. No subprocesses, no
parallelism - launching 30 processes at once nearly filled the machine's RAM
last time (see CLAUDE.md), and a plain sequential loop is a lot easier to
reason about anyway. Just takes longer.
"""

from train_gesture_model import train as train_gesture_model
from train_gesture_temporal_model import train as train_gesture_temporal_model

input_types = ["bandpower", "csp", "dwt"]
k = 5

results = {}

for input_type in input_types:
    for fold in range(k):
        val_acc = train_gesture_model(input_type=input_type, fold=fold)
        results[f"gesture_model/{input_type}/fold{fold}"] = val_acc

for input_type in input_types:
    for fold in range(k):
        val_acc = train_gesture_temporal_model(input_type=input_type, fold=fold)
        results[f"gesture_temporal_model/{input_type}/fold{fold}"] = val_acc

print("\nsweep done, final val_acc per run:")
for name, val_acc in results.items():
    print(f"{name}: {val_acc:.3f}")
