import numpy as np
import pyarrow as pa
from dora import Node
import mediapipe as mp
from eeg.data_collection import JointData, Joint, DataType

mp_hands = mp.solutions.hands


def process_input(joint_data: JointData, time_step: int):
    print(time_step)
    res = None

    origin = joint_data.get_positions(DataType.NORM, Joint.W)[time_step]
    mid_mcp = joint_data.get_positions(DataType.NORM, Joint.MM)[time_step]
    pinky_mcp = joint_data.get_positions(DataType.NORM, Joint.PM)[time_step]

    unit_z = mid_mcp - origin  # vector from wrist to mid_mcp
    # divide by magnitude (becomes unit vector)
    unit_z = unit_z/np.linalg.norm(unit_z)

    vec_y = pinky_mcp - origin
    unit_x = np.cross(vec_y, unit_z)  # perpendicular
    unit_x = unit_x/np.linalg.norm(unit_x)

    unit_y = np.cross(unit_z, unit_x)

    # rotational matrix for linear transformation
    R: np.ndarray = np.array([-unit_x, -unit_y, unit_z]).reshape((3, 3))

    def change_of_basis(tip_idx: Joint, mcp_idx: Joint) -> np.ndarray:
        v = joint_data.get_positions(DataType.WORLD, tip_idx)[time_step] - \
            joint_data.get_positions(DataType.WORLD, mcp_idx)[time_step]
        return R @ v

    index = change_of_basis(Joint.IT, Joint.IM)
    middle = change_of_basis(Joint.MT, Joint.MM)
    ring = change_of_basis(Joint.RT, Joint.RM)
    thumb = change_of_basis(Joint.TT, Joint.TM)

    print("index:", index)
    print("middle:", middle)
    print("ring:", ring)
    print("thumb:", thumb)
    print()

    res = [{'r_tip1': index, 'r_tip2': middle,
            'r_tip3': ring, 'r_tip4': thumb}]

    return res


def main():
    node = Node()

    joint_data = JointData(
        "C:\\Users\\prabh\\Documents\\eeg\\data\\test2.npy")

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
