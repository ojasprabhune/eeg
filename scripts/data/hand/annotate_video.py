import argparse
import os
import warnings

import absl.logging
import cv2
import numpy as np

from eeg.eeg_data.datasets.utils import Colors

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

absl.logging.set_verbosity(absl.logging.ERROR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hides INFO + WARNING


# --- ARGPARSE ---
parser = argparse.ArgumentParser(description="video labeler: press 1-4 to label frames")
parser.add_argument("video_path", type=str, help="path to the input video file")
parser.add_argument("output_npy", type=str, help="path to save the output .npy file")
args = parser.parse_args()

video_path = args.video_path
output_npy = args.output_npy

# --- LABEL MAP ---
label_map = {1: "fist", 2: "left", 3: "fingers", 4: "open"}

# --- LOAD VIDEO ---
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
speed = 1
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# initialize labels
labels = np.ones(total_frames)
current_label = 4
frame_idx = 0

# --- PLAYBACK LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # draw label text
    text = f"{label_map[current_label]} {current_label}"
    time_left_sec = str(int((total_frames - frame_idx) / cap.get(cv2.CAP_PROP_FPS)))
    diagnostic1 = "press 1-4 to label gestures"
    diagnostic2 = "'s' to toggle 2x speed, 'p' to pause"
    diagnostic3 = "'d' to go back 5s, 'q' to quit"
    diagnostic4 = f"current label: {text}, time left: {time_left_sec}s, speed: {speed}x"

    # draw each line separately
    y0, dy = 50, 30  # starting y and line spacing
    for i, line in enumerate([diagnostic1, diagnostic2, diagnostic3, diagnostic4]):
        y = y0 + i * dy
        color = (
            (0, 255, 0) if i == 3 else (255, 255, 255)
        )  # highlight current label line
        cv2.putText(
            frame,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    # show frame
    cv2.imshow("video labeler", frame)

    # write label for this frame
    labels[frame_idx] = current_label
    frame_idx += 1

    delay = int(35 / speed)

    # keyboard input
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("s"):
        speed = 2 if speed == 1 else 1
    elif key == ord("1"):
        current_label = 1
    elif key == ord("2"):
        current_label = 2
    elif key == ord("3"):
        current_label = 3
    elif key == ord("4"):
        current_label = 4
    elif key == ord("p"):
        # pause until 'p' is pressed again
        while True:
            key2 = cv2.waitKey(30) & 0xFF
            if key2 == ord("p"):
                break
    elif key == ord("d"):
        # go back 5 seconds
        frame_idx = max(0, frame_idx - int(5 * cap.get(cv2.CAP_PROP_FPS)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    elif key == ord("q"):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()

labels = labels[:frame_idx]  # trim in case of early quit
np.save(output_npy, labels)
print(f"{Colors.OKGREEN}SUCCESS! saved labels to {output_npy}{Colors.ENDC}")
