import re
from collections import Counter, defaultdict

from eeg.gesture2hand import gesture_experiments, get_gesture_class
from eeg.language_model.language_dataset import load_experiment_corpus


def make_distribution(experiment: str, sample_sentences: int = 70):
    raw_sentences = load_experiment_corpus("eeg/language_model/data/", experiment)

    sentences = [re.sub(r"[^a-z]", "", sentence.lower()) for sentence in raw_sentences]

    sentences = sentences[:sample_sentences]
    counts = Counter("".join(sentences))

    scale = sample_sentences / len(sentences)

    distribution = {}
    sampled_letters = []

    print("\n" + "=" * 55)
    print(f"{experiment.upper()}")
    print("=" * 55)
    print(f"Original sentences: {len(sentences)}")
    print(f"Sample sentences:   {sample_sentences}")
    print()

    print(f"{'Letter':<10}{'Original':>12}{'Sampled':>12}")
    print("-" * 36)

    for letter in gesture_experiments[experiment]:
        original = counts.get(letter, 0)
        sampled = round(original * scale)

        distribution[letter] = sampled
        sampled_letters.extend([letter] * sampled)

        print(f"{letter.upper():<10}{original:>12}{sampled:>12}")

    print(f"\nTotal trials: {sum(distribution.values())}")

    # gesture class distribution
    class_counts = Counter(
        get_gesture_class(letter, experiment=experiment, zero_based_idx=False)
        for letter in sampled_letters
    )

    print("\nCLASS DISTRIBUTION")
    print("-" * 36)
    print(f"{'Class':<10}{'Trials':>12}{'Percent':>12}")

    total_classes = sum(class_counts.values())

    for gesture_class in sorted(class_counts):
        count = class_counts[gesture_class]
        percent = (count / total_classes) * 100

        print(f"{gesture_class:<10}{count:>12}{percent:>11.2f}%")

    return distribution


def summarize_all(
    experiments: list[str], sample_sentences: int = 70, trial_length: int = 8
):
    total_distribution = defaultdict(int)
    total_trials = 0

    total_class_distribution = defaultdict(int)

    for experiment in experiments:
        distribution = make_distribution(experiment, sample_sentences)

        for letter, count in distribution.items():
            total_distribution[letter] += count

        # collect combined gesture classes
        for letter, count in distribution.items():
            gesture_class = get_gesture_class(
                letter, experiment=experiment, zero_based_idx=False
            )
            total_class_distribution[gesture_class] += count

        total_trials += sum(distribution.values())

    print("\n" + "=" * 55)
    print("COMBINED LETTER REPETITIONS")
    print("=" * 55)

    print(f"{'Letter':<10}{'Repetitions':>15}")
    print("-" * 30)

    for letter in sorted(total_distribution):
        print(f"{letter.upper():<10}{total_distribution[letter]:>15}")

    print("\n" + "=" * 55)
    print("COMBINED CLASS DISTRIBUTION")
    print("=" * 55)

    print(f"{'Class':<10}{'Trials':>12}{'Percent':>12}")
    print("-" * 36)

    for gesture_class in sorted(total_class_distribution):
        count = total_class_distribution[gesture_class]
        percent = (count / total_trials) * 100

        print(f"{gesture_class:<10}{count:>12}{percent:>11.2f}%")

    hours = (total_trials * trial_length) / 3600

    print("\n" + "=" * 55)
    print("DATA COLLECTION ESTIMATE")
    print("=" * 55)
    print(f"Total trials:       {total_trials}")
    print(f"Trial duration:     {trial_length} seconds")
    print(f"Total time:         {hours:.2f} hours")


summarize_all(
    ["asl_8_letters", "common_8_letters", "6_letters"],
    sample_sentences=50,
    trial_length=8,
)
