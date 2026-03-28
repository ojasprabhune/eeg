import mne
import argparse
import numpy as np

from eeg.eeg_data.datasets.utils import Colors

parser = argparse.ArgumentParser(
    prog="Trim EEG and hand position data",
    description="Trim EEG and hand position data to specific times",
)
parser.add_argument("eeg_data_path", type=str)
parser.add_argument("hand_data_path", type=str)
parser.add_argument("eeg_trim_start_seconds", type=float)
parser.add_argument("hand_trim_start_seconds", type=float)
parser.add_argument("start_offset_seconds", type=float)

args = parser.parse_args()

eeg_data_path: str = args.eeg_data_path
hand_data_path: str = args.hand_data_path
fps = 29.973234

if eeg_data_path.endswith(".edf") or hand_data_path.endswith(".npy"):
    print(f"{Colors.FAIL}Error: Paths should not include file extensions.{Colors.ENDC}")
    quit()


def trim_recording(
    eeg_data_path: str,
    hand_data_path: str,
    eeg_trim_start_seconds: float,
    hand_trim_start_seconds: float,
    start_offset_seconds: float,
    output_edf_path=None,
    output_npy_path=None,
) -> None:

    # --- EEG ---
    print(f"{Colors.OKBLUE}Reading EDF...{Colors.ENDC}")
    raw: mne.io.Raw = mne.io.read_raw_edf(f"{eeg_data_path}.edf", preload=True)
    start_seconds = eeg_trim_start_seconds + start_offset_seconds
    raw_trimmed: mne.io.Raw = raw.copy().crop(tmin=start_seconds)
    print(f"{Colors.OKBLUE}Trimmed EDF.{Colors.ENDC}\n")

    # --- hand ---
    print(f"{Colors.OKCYAN}Reading NPY...{Colors.ENDC}")
    hand_data = np.load(f"{hand_data_path}.npy")
    start_frame = int((hand_trim_start_seconds + start_offset_seconds) * fps)
    hand_trimmed = hand_data[:, start_frame:, :]
    print(f"{Colors.OKCYAN}Trimmed NPY.{Colors.ENDC}\n")

    initial_eeg_length = raw.duration
    initial_hand_length = hand_data.shape[1] / fps

    eeg_length = raw_trimmed.duration
    hand_length = hand_trimmed.shape[1] / fps

    # --- trim to same length ---
    print(f"{Colors.OKGREEN}Trimming data ends...{Colors.ENDC}\n")
    min_length = min(eeg_length, hand_length)
    raw_trimmed.crop(tmin=0, tmax=min_length)
    n_frames = int(min_length * fps)
    hand_trimmed = hand_trimmed[:, :n_frames, :]

    raw_trimmed.save(f"{eeg_data_path}_cut_raw.fif", overwrite=True)
    print(f"{Colors.OKBLUE}Saved EDF.{Colors.ENDC}\n")
    np.save(f"{hand_data_path}_cut.npy", hand_trimmed)
    print(f"{Colors.OKCYAN}Saved NPY.{Colors.ENDC}\n")

    # --- stats ---
    final_eeg_length = raw_trimmed.duration
    final_hand_length = hand_trimmed.shape[1] / fps

    print(f"EEG data shape: {Colors.BOLD}{raw.get_data().shape}{Colors.ENDC}")
    print(f"Hand data shape: {Colors.BOLD}{hand_data.shape}{Colors.ENDC}\n")

    print(
        f"{Colors.OKCYAN}EEG data length before: {round(initial_eeg_length, 1)} seconds.{Colors.ENDC}"
    )
    print(
        f"{Colors.OKCYAN}Hand data length before: {round(initial_hand_length, 1)} seconds.{Colors.ENDC}\n"
    )

    print(
        f"{Colors.OKGREEN}EEG data length after: {round(final_eeg_length, 1)} seconds.{Colors.ENDC}"
    )
    print(
        f"{Colors.OKGREEN}Hand data length after: {round(final_hand_length, 1)} seconds.{Colors.ENDC}\n"
    )

    print("start seconds EEG:", start_seconds)
    print("start seconds hand:", start_frame / fps, "\n")


trim_recording(
    eeg_data_path,
    hand_data_path,
    eeg_trim_start_seconds=args.eeg_trim_start_seconds,
    hand_trim_start_seconds=args.hand_trim_start_seconds,
    start_offset_seconds=args.start_offset_seconds,
)

