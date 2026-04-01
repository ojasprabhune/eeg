import argparse
import numpy as np
import pyarrow as pa
from dora import Node
import mediapipe as mp
from eeg.data_collection import JointData, Joint, DataType
from eeg.big_hand.position_llm.utils import appendages

mp_hands = mp.solutions.hands


def process_input(joint_data: JointData, time_step: int):
    result = appendages(joint_data)

    if time_step >= len(result):
        return None

    res = [
        {
            "r_tip1": result[time_step, 0:3],
            "r_tip2": result[time_step, 3:6],
            "r_tip3": result[time_step, 6:9],
            "r_tip4": result[time_step, 9:12],
        }
    ]

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        help="Path to the .npy hand positions file",
        required=True,
    )
    args = parser.parse_args()

    node = Node()

    joint_data = JointData(args.data_file)

    pa.array([])  # initialize pyarrow array
    time_step = 0

    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if event_id == "tick":
                # process
                res = process_input(joint_data, time_step)

                if res is not None:
                    node.send_output("r_hand_pos", pa.array(res))

                time_step += 1

        elif event_type == "ERROR":
            raise RuntimeError(event["error"])


if __name__ == "__main__":
    main()
