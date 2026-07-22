"""
Filter a manually-written corpus set: de-duplicate, keep only sentences whose
words are all in the experiment vocab (words.txt), and cap at 500. Writes the
filtered corpus next to the source and prints a summary.

The same filtering runs automatically inside LanguageDataset (via
load_experiment_corpus), so training always sees the filtered corpus; this
script is just for inspecting/exporting the result manually.
"""

import os

from eeg.language_model.language_dataset import load_experiment_corpus

DATA_DIR = "eeg/language_model/data"


def filter_sentence(experiment: str) -> None:
    filtered = load_experiment_corpus(DATA_DIR, experiment)

    # write next to the source set for inspection
    exp_dir = os.path.join(DATA_DIR, experiment)
    sets = [n for n in os.listdir(exp_dir) if n.startswith("set_") and n[4:].isdigit()]
    latest = max(sets, key=lambda n: int(n[4:]))
    out_path = os.path.join(exp_dir, latest, f"{experiment}_filtered.txt")
    with open(out_path, "w") as file:
        file.write("\n".join(filtered) + "\n")

    print(f"=== Experiment: {experiment} ===")
    print(f"Filtered: {len(filtered)}  ->  {out_path}\n")


if __name__ == "__main__":
    filter_sentence(experiment="asl_8_letters")
    filter_sentence(experiment="common_8_letters")
    filter_sentence(experiment="6_letters")
