import cv2
import numpy as np
import pyarrow as pa
from dora import Node
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def process_img(hand_proc, image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand_proc.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    r_res = None
    l_res = None
    if results.multi_hand_landmarks:
        for index, handedness_classif in enumerate(results.multi_handedness):
            # let's considere only one right hand
            if handedness_classif.classification[0].score > 0.8:
                # metric
                hand_landmarks = results.multi_hand_world_landmarks[index]
                # hand_landmarks=results.multi_hand_landmarks[index] #normalized
                # normalized
                hand_landmarks_norm = results.multi_hand_landmarks[index]

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

                # rotate everything in a hand referential
                origin = np.array(
                    [
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].x,
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].y,
                        # wrist base as the origin
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].z,
                    ]
                )
                mid_mcp = np.array(
                    [
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].x,
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].y,
                        # base of the middle finger
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].z,
                    ]
                )
                # z is unit vector from base of wrist toward base of middle finger
                unit_z = mid_mcp - origin
                unit_z = unit_z / np.linalg.norm(unit_z)
                pinky_mcp = np.array(
                    [
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                        # base of the pinky finger
                        hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].z,
                    ]
                )

                index_mcp = np.array(
                    [
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].x,
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].y,
                        # base of the index finger
                        hand_landmarks_norm.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].z,
                    ]
                )

                vec_towards_y = np.array([0, 0, 0])
                if handedness_classif.classification[0].label == "Right":
                    vec_towards_y = (
                        pinky_mcp - origin
                    )  # vector from wrist base towards pinky base
                if handedness_classif.classification[0].label == "Left":
                    vec_towards_y = (
                        index_mcp - origin
                    )  # vector from wrist base towards pinky base

                # we say unit x is the cross product of z and the vector towards pinky
                unit_x = np.cross(vec_towards_y, unit_z)

                unit_x = unit_x / np.linalg.norm(unit_x)

                unit_y = np.cross(unit_z, unit_x)

                R = np.ndarray([])
                if handedness_classif.classification[0].label == "Right":
                    # -y because of mirror?
                    R = np.array([unit_x, -unit_y, unit_z]).reshape((3, 3))
                if handedness_classif.classification[0].label == "Left":
                    # -y because of mirror?
                    R = np.array([unit_x, -unit_y, unit_z]).reshape((3, 3))

                tip1 = R @ np.array([tip1_x, tip1_y, tip1_z])
                tip2 = R @ np.array([tip2_x, tip2_y, tip2_z])
                tip3 = R @ np.array([tip3_x, tip3_y, tip3_z])
                tip4 = R @ np.array([tip4_x, tip4_y, tip4_z])

                if handedness_classif.classification[0].label == "Right":
                    r_res = [
                        {"r_tip1": tip1, "r_tip2": tip2, "r_tip3": tip3, "r_tip4": tip4}
                    ]
                elif handedness_classif.classification[0].label == "Left":
                    l_res = [
                        {"l_tip1": tip1, "l_tip2": tip2, "l_tip3": tip3, "l_tip4": tip4}
                    ]

    return image, r_res, l_res


def main():

    node = Node()

    pa.array([])  # initialize pyarrow array
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        for event in node:
            event_type = event["type"]

            if event_type == "INPUT":
                event_id = event["id"]

                if event_id == "tick":
                    ret, frame = cap.read()

                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    # process
                    frame, r_res, l_res = process_img(hands, frame)

                    if r_res is not None:
                        node.send_output("r_hand_pos", pa.array(r_res))
                    if l_res is not None:
                        node.send_output("l_hand_pos", pa.array(l_res))
                    # cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
                    cv2.imshow("MediaPipe Hands", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            elif event_type == "ERROR":
                raise RuntimeError(event["error"])


if __name__ == "__main__":
    main()
