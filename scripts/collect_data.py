import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

from eeg.overfit.data_collection import JointData

parser = argparse.ArgumentParser(
    prog="Collect data",
    description="Run hand pose detection and obtain joint position deltas",
)
parser.add_argument("filename", type=str, help="The file name of the npy data")
parser.add_argument(
    "data_time", type=int, help="Length of hand pose estimation data in seconds"
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
parser.add_argument("--joint", type=str, default="W", help="Joint data to plot")
args = parser.parse_args()

# each joint is a landmark
mp_drawing = mp.solutions.drawing_utils  # helps draw landmarks on screen
mp_hands = mp.solutions.hands  # hands model

joint_data = []  # initialize list of frames with landmarks
filename = args.filename


def hand_detection(mp_hands, mp_drawing, joint_data: list, set_fps: int):
    # getting camera / webcam (0, 1, 2 for connected webcams)
    cap = cv2.VideoCapture(0)

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
            frame_data = []

            if not success:
                print("Ignoring empty camera frame.")
                # if loading a video, use 'break' instead of 'continue'.
                continue

            # to improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )  # model requires 3 channel RGB
            results = hands.process(frame)

            # set flag back to true
            # allows us to draw on image and render
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            image_height, image_width, _ = frame.shape

            # if solution has landmarks then render results on image
            if results.multi_hand_landmarks:
                # iterate through each landmark
                # hand_landmarks represents the landmarks for that hand
                for hand_landmarks in results.multi_hand_landmarks:
                    # pass in three variables:
                    # 1. image
                    # 2. hand (set of landmarks)
                    # 3. HAND_CONNECTIONS represents the set of coordinates of relations between joints
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # iterate through hand landmarks list and figure out the position
                    # openCV origin is top left (e.g., y position increases when wrist moves down)
                    for landmark in hand_landmarks.landmark:
                        W = hand_landmarks.landmark[0]
                        relative_x = landmark.x - W.x
                        relative_y = landmark.y - W.y
                        relative_z = landmark.z - W.z
                        # extend list by new landmark positions
                        frame_data.extend(
                            [
                                relative_x * image_width,
                                relative_y * image_height,
                                relative_z * image_width,
                            ]
                        )

                # add to data
                joint_data.append(frame_data)

            # render image to screen using OpenCV with "Hand tracking" window title
            cv2.imshow("Hand tracking", frame)

            if args.save_video:
                writer.write(frame)

            # hit "q" and close window
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            if time.time() > data_time:
                break

    # release webcam
    cap.release()
    # close windows
    cv2.destroyAllWindows()


def write(data: list, filename: str):
    dataset = np.array(data)  # turn list into numpy array
    np.save(f"data/{filename}.npy", dataset)  # save numpy array


hand_detection(mp_hands, mp_drawing, joint_data, 30)
write(joint_data, filename)

data = JointData(f"data/{filename}.npy")
if args.plot:
    data.plot_data(args.joint)
