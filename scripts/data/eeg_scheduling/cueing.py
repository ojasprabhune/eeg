"""
An EEG cueing script that uses LSL streaming to sync with EEG data. The script
presents gesture cues of the letters of each experiment corpus with fixed
timing while sending event markers through an LSL stream for synchronization
with EEG recording. It outputs visual and audio cues and LSL event markers.
"""

import time

from pylsl import StreamInfo, StreamOutlet, cf_string
from pylsl.util import IRREGULAR_RATE

from eeg.gesture2hand import Colors
from eeg.gesture2hand.utils.gestures import get_gesture_class
from eeg.language_model.language_dataset import LanguageDataset
from eeg.language_model.tokenizer import LanguageTokenizer


def num_classes_for(experiment: str) -> int:
    return 6 if experiment == "6_letters" else 4


# ==========================
# CONFIGURATION
# ==========================

EXPERIMENT = "6_letters"
GESTURE_TYPE = "hand"
NUM_SENTENCES = 40
START_TRIALS = 1

GESTURE_LETTERS = [
    "A",
    "B",
    "C",
    "D",
    "E",
]

TRIALS_PER_GESTURE = 1

REST_TIME = 2.0  # before cue
CUE_TIME = 2.0  # cue displayed
PRE_EXEC = 1.0  # prevent mcrps from interfereing with motor execution
MOTOR_EXEC = 3.0  # motor execution

# ==========================
# LSL SETUP
# ==========================

info = StreamInfo(
    name="EEGMarkers",
    type="Markers",
    channel_count=1,
    nominal_srate=IRREGULAR_RATE,
    channel_format=cf_string,
    source_id="gesture_marker_stream",
)

outlet = StreamOutlet(info)

tokenizer = LanguageTokenizer()

# ==========================
# BUILD TRIAL LIST
# ==========================


langage_dataset = LanguageDataset(
    num_classes=num_classes_for(EXPERIMENT),
    experiment=EXPERIMENT,
)

number_gesture_map = {}

if GESTURE_TYPE == "hand":
    if EXPERIMENT == "asl_8_letters":
        number_gesture_map = {
            1: "Fist",
            2: "Left",
            3: "Fingers",
            4: "Open",
        }
    elif EXPERIMENT == "common_8_letters":
        number_gesture_map = {
            1: "Fist, thumb pointing up between fingers",
            2: "Fist, thumb curled in front",
            3: "Extraneous thumb, medial fingers curled",
            4: "Medial fingers point up, thumb inside palm",
        }
    elif EXPERIMENT == "6_letters":
        number_gesture_map = {
            1: "Fist, thumb curled under fingers",
            2: "Fist, thumb between index and middle fingers",
            3: "Fist, thumb pointing up on the side",
            4: "Fingers curled, index and thumb touching",
            5: "Fist, pinky up",
            6: "Fist, thumb between middle and ring fingers",
        }

else:
    if EXPERIMENT in ["asl_8_letters", "common_8_letters"]:
        number_gesture_map = {
            1: "Right-arm, gross, upward: Tree",
            2: "Left-arm, gross, upward: Tree (left-dominant)",
            3: "Bilateral, symmetric, expanding: Big",
            4: "Bilateral + non-arm effector: Rain",
        }
    elif EXPERIMENT == "6_letters":
        number_gesture_map = {
            1: "Right-arm, gross, upward: Tree",
            2: "Left-arm, gross, upward: Tree (left-dominant)",
            3: "Bilateral, symmetric, expanding: Big",
            4: "Bilateral + non-arm effector: Rain",
            5: "Bilateral, gross, pointing forward: Go",
            6: "Right-arm, gross, beckoning: Come",
        }

sentences = langage_dataset.sample_train_sentences(num_sentences=NUM_SENTENCES)
gesture_sentences = [
    [
        get_gesture_class(letter, experiment=EXPERIMENT, zero_based_idx=False)
        for letter in sentence
    ]
    for sentence in sentences
]


total_trials = sum(1 for sentence in gesture_sentences for letter in sentence)

print(f"\nTotal trials: {total_trials}")
input("Press ENTER to begin...\n")

# ==========================
# CUEING LOOP
# ==========================

trial_count = 1

for i, sentence in enumerate(gesture_sentences):
    print(f"\n{Colors.OKBLUE}Sentence: {sentences[i]}{Colors.ENDC}")

    for j, letter in enumerate(sentence):
        if trial_count < START_TRIALS:
            continue

        print(f"\n-------- Trial {trial_count}/{total_trials} --------")

        # rest
        print(f"{Colors.OKGREEN}> REST{Colors.ENDC}")
        outlet.push_sample(["REST_START"])
        time.sleep(REST_TIME)

        # cue
        print(
            f"> TARGET LETTER: {Colors.BOLD}{Colors.OKCYAN}{sentences[i][j]}{Colors.ENDC}"
        )
        print(
            f"> TARGET GESTURE: {Colors.BOLD}{Colors.OKCYAN}{number_gesture_map[letter]}{Colors.ENDC}"
        )
        outlet.push_sample([f"CUE_{letter}"])
        time.sleep(CUE_TIME)

        # pre movement
        print(f"{Colors.WARNING}> REST{Colors.ENDC}")
        outlet.push_sample([f"PRE_{letter}"])
        time.sleep(PRE_EXEC)

        # movement
        print(f"{Colors.BOLD}{Colors.HEADER}> GO 3{Colors.ENDC}")
        outlet.push_sample([f"MOVE_{letter}"])
        time.sleep(MOTOR_EXEC)

        trial_count += 1

    print(f"\n{Colors.OKBLUE}Sentence finished.{Colors.ENDC}")
    outlet.push_sample(["END_SEQ"])

print(f"\nExperiment {EXPERIMENT} complete.")
input("Press ENTER to finish...")
outlet.push_sample(["SESSION_END"])
