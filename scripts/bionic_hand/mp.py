import argparse
import os
import time

from eeg.region_token.data_collection import Joint, JointData

import cv2
import numpy as np
import pyarrow as pa
from dora import Node
import mediapipe as mp
from scipy.spatial.transform import Rotation

from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def update_frame(frame, scatter_plot, joint_data: JointData):
    print(frame)
    x_data = []
    y_data = []
    for joint in Joint:
        x_data.append(joint_data.get_positions(joint)[int(frame), 0])
        y_data.append(-joint_data.get_positions(joint)[int(frame), 1])

    x_data, y_data = change_of_basis(frame, joint_data)

    scatter_plot.set_offsets(np.c_[x_data, y_data])
    return (scatter_plot,)


def change_of_basis(frame: int, joint_data: JointData):
    def get_pos(joint: Joint, component: str):
        return joint_data.get_positions(joint, component)[frame]

    tip1_x = get_pos(Joint.IT, "x") - get_pos(Joint.IM, "x")
    tip1_y = get_pos(Joint.IT, "y") - get_pos(Joint.IM, "y")
    tip1_z = get_pos(Joint.IT, "z") - get_pos(Joint.IM, "z")

    tip2_x = get_pos(Joint.MT, "x") - get_pos(Joint.MM, "x")
    tip2_y = get_pos(Joint.MT, "y") - get_pos(Joint.MM, "y")
    tip2_z = get_pos(Joint.MT, "z") - get_pos(Joint.MM, "z")

    tip3_x = get_pos(Joint.RT, "x") - get_pos(Joint.RM, "x")
    tip3_y = get_pos(Joint.RT, "y") - get_pos(Joint.RM, "y")
    tip3_z = get_pos(Joint.RT, "z") - get_pos(Joint.RM, "z")

    tip4_x = get_pos(Joint.TT, "x") - get_pos(Joint.TM, "x")
    tip4_y = get_pos(Joint.TT, "y") - get_pos(Joint.TM, "y")
    tip4_z = get_pos(Joint.TT, "z") - get_pos(Joint.TM, "z")

    return [tip1_x, tip2_x, tip3_x, tip4_x], [-tip1_y, -tip2_y, -tip3_y, -tip4_y]


def main():
    data = JointData("data/open_peace_front.npy")

    fig, ax = plt.subplots()
    x_data, y_data = [], []
    scatter_plot = ax.scatter(x_data, y_data)

    ax.set_xlim(-300, 300)
    ax.set_ylim(-100, 400)
    ax.set_title("Streaming Scatter Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    ani = FuncAnimation(
        fig,
        partial(update_frame, scatter_plot=scatter_plot, joint_data=data),
        frames=range(len(data.get_dataset())),
        blit=True,
        interval=33,
    )

    plt.show()


def process_img(hand_proc, image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand_proc.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    r_res = None
    l_res = None
    if results.multi_hand_landmarks:
        for index, handedness_classif in enumerate(results.multi_handedness):
            if handedness_classif.classification[0].score > 0.8:
                hand_landmarks = results.multi_hand_world_landmarks[index]  # metric
                hand_landmarks_norm = results.multi_hand_landmarks[index]  # normalized

                tip1_x = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                )
                tip1_y = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                )
                tip1_z = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z
                    - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z
                )

                tip2_x = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                )
                tip2_y = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                    - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                )
                tip2_z = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
                    - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
                )

                tip3_x = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
                )
                tip3_y = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                    - hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
                )
                tip3_z = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
                    - hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z
                )

                tip4_x = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
                )
                tip4_y = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                    - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
                )
                tip4_z = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z
                    - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z
                )

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks_norm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                origin = np.array(
                    [
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].x,
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].y,
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].z,
                    ]
                )  # wrist base as the origin
                mid_mcp = np.array(
                    [
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].x,
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].y,
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].z,
                    ]
                )  # base of the middle finger
                unit_z = (
                    mid_mcp - origin
                )  # z is unit vector from base of wrist toward base of middle finger
                unit_z = unit_z / np.linalg.norm(unit_z)
                pinky_mcp = np.array(
                    [
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].z,
                    ]
                )  # base of the pinky finger

                index_mcp = np.array(
                    [
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].x,
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].y,
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].z,
                    ]
                )  # base of the index finger

                if handedness_classif.classification[0].label == "Right":
                    vec_towards_y = (
                        pinky_mcp - origin
                    )  # vector from wrist base towards pinky base
                if handedness_classif.classification[0].label == "Left":
                    vec_towards_y = (
                        index_mcp - origin
                    )  # vector from wrist base towards pinky base

                unit_x = np.cross(
                    vec_towards_y, unit_z
                )  # we say unit x is the cross product of z and the vector towards pinky

                unit_x = unit_x / np.linalg.norm(unit_x)

                unit_y = np.cross(unit_z, unit_x)

                if handedness_classif.classification[0].label == "Right":
                    R = np.array([unit_x, -unit_y, unit_z]).reshape(
                        (3, 3)
                    )  # -y because of mirror?
                if handedness_classif.classification[0].label == "Left":
                    R = np.array([unit_x, -unit_y, unit_z]).reshape(
                        (3, 3)
                    )  # -y because of mirror?
                tip1 = R @ np.array([tip1_x, tip1_y, tip1_z])
                tip2 = R @ np.array([tip2_x, tip2_y, tip2_z])
                tip3 = R @ np.array([tip3_x, tip3_y, tip3_z])
                tip4 = R @ np.array([tip4_x, tip4_y, tip4_z])

                if handedness_classif.classification[0].label == "Right":
                    r_res = [
                        {"r_tip1": tip1, "r_tip2": tip2, "r_tip3": tip3, "r_tip4": tip4}
                    ]
                    print(
                        f"RIGHT: {tip1_x:.3f} {tip1_y:.3f} {tip1_z:.3f} => {tip1}. {unit_x} {unit_y} {unit_z}"
                    )
                elif handedness_classif.classification[0].label == "Left":
                    l_res = [
                        {"l_tip1": tip1, "l_tip2": tip2, "l_tip3": tip3, "l_tip4": tip4}
                    ]
                    print(
                        f"LEFT: {tip1_x:.3f} {tip1_y:.3f} {tip1_z:.3f} => {tip1}. {unit_x} {unit_y} {unit_z}"
                    )

    # Flip the image horizontally for a selfie-view display.
    return image, r_res, l_res


# cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))


def old_main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            # process
            frame, r_res, l_res = process_img(hands, frame)

            # cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
            cv2.imshow("MediaPipe Hands", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    old_main()
