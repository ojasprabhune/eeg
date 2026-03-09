import numpy as np
import pyarrow as pa
from dora import Node
import mediapipe as mp
from eeg.data_collection import JointData, Joint, DataType
from eeg.big_hand.position_llm.utils import appendages

mp_hands = mp.solutions.hands


def process_input(joint_data: JointData, time_step: int):
    result = appendages(joint_data)

    res = [{'r_tip1': result[time_step, 0:3], 'r_tip2': result[time_step, 3:6],
            'r_tip3': result[time_step, 6:9], 'r_tip4': result[time_step, 9:12]}]

    return res


def main():
    node = Node()

    joint_data = JointData(
        "C:/Users/prabh/Documents/eeg/data/open_fist.npy")

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
                    node.send_output('r_hand_pos', pa.array(res))

        elif event_type == "ERROR":
            raise RuntimeError(event["error"])

        time_step += 1


if __name__ == "__main__":
    main()
