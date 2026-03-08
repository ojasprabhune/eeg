import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

from eeg.data_collection import JointData

parser = argparse.ArgumentParser(
    prog="Collect data",
    description="Run hand pose detection and obtain joint position deltas",
)
parser.add_argument("filename", type=str, help="The file name of the npy data")
parser.add_argument(
    "data_time", type=int, help="Length of hand pose estimation data in seconds"
)
parser.add_argument(
    "--webcam",
    type=bool,
    default=True,
    help="Either run on webcam video or video file"
)
parser.add_argument(
    "--input_video",
    type=str,
    default="",
    help="Video file to run estimation on (if selected)"
)
parser.add_argument(
    "--save_video",
    type=bool,
    default=False,
    help="Saves a video of hand pose estimation if True",
)
parser.add_argument(
    "--plot", type=bool, default=False, help="Plots the results if True"
)
parser.add_argument("--joint", type=str, default="W",
                    help="Joint data to plot")

args = parser.parse_args()

# each joint is a landmark
mp_drawing = mp.solutions.drawing_utils  # helps draw landmarks on screen
mp_drawing_styles = mp.solutions.drawing_styles  # drawings are colorful
mp_hands = mp.solutions.hands  # hands model

joint_data_world = []  # initialize list of frames with world landmarks
joint_data_norm = []  # initialize list of frames with norm landmarks
filename = args.filename
num_frames = args.data_time * 30
input_video_path = args.input_video


def hand_detection(mp_hands, mp_drawing, joint_data_world: list, joint_data_norm: list, set_fps: int):
    cur_frame = 1

    # getting camera / webcam (0, 1, 2 for connected webcams)
    if args.webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_video_path)

    # set desired FPS to camera
    cap.set(cv2.CAP_PROP_FPS, set_fps)
    # verify FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    # log FPS for verification
    print(f"Set FPS: {set_fps}\nFPS: {fps}")

    data_time = args.data_time + time.time()

    if args.save_video:
        path = f"data/videos/{filename}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    # resource management: open hands as the term below and close it automatically
    # two metrics:
    # 1. detection: threshold for initial detection to be successful
    # 2. tracking: threshold for tracking after initial detection
    with mp_hands.Hands(
        min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1
    ) as hands:
        # reading frames while the capture is opened
        while cap.isOpened():
            # read each frame from webcam
            # ret and frame variables unpacking cap.read() function
            # a return value and image from webcame
            success, frame = cap.read()
            # initialize list for current frame data
            frame_data_world = []
            frame_data_norm = []

            if not success:
                print("Ignoring empty camera frame.")
                # if loading a video, use 'break' instead of 'continue'.
                continue

            # to improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False

            # model requires 3 channel RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame)

            # set flag back to true
            # allows us to draw on image and render
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # metric
                hand_landmarks = results.multi_hand_world_landmarks[0]

                # normalized
                hand_landmarks_norm = results.multi_hand_landmarks[0]

                # extend lists by new landmark positions
                for world_landmark in hand_landmarks.landmark:
                    frame_data_world.extend(
                        [world_landmark.x, world_landmark.y, world_landmark.z])

                for norm_landmark in hand_landmarks_norm.landmark:
                    frame_data_norm.extend(
                        [norm_landmark.x, norm_landmark.y, norm_landmark.z])

                # add time step to data
                joint_data_world.append(frame_data_world)
                joint_data_norm.append(frame_data_norm)
                cur_frame += 1

            # render image to screen using OpenCV with "Hand tracking" window title
            cv2.imshow("Hand tracking", frame)

            if args.save_video:
                writer.write(frame)

            # hit "q" and close window
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            if cur_frame > num_frames:
                break

    # release webcam
    cap.release()
    # close windows
    cv2.destroyAllWindows()


def write(data: list, filename: str):
    dataset = np.array(data)  # turn lists into numpy array
    np.save(f"data/{filename}.npy", dataset)  # save numpy array


hand_detection(mp_hands, mp_drawing, joint_data_world, joint_data_norm, 30)
write([joint_data_world, joint_data_norm], filename)

data = JointData(f"data/{filename}.npy")
if args.plot:
    if args.joint is not None:
        data.plot_data(args.joint)
    else:
        print("Error: No joint given to plot.")
