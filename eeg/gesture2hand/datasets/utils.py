import numpy as np
import os

from eeg.data_collection import JointData, Joint, DataType


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def appendages(joint_data: JointData) -> np.ndarray:
    """
    Uses joint data to calculate appendage vectors and
    unit vectors to perform change of basis and return
    an numpy array of size (T, 12).
    """
    origin = joint_data.get_positions(DataType.NORM, Joint.W)
    mid_mcp = joint_data.get_positions(DataType.NORM, Joint.MM)
    pinky_mcp = joint_data.get_positions(DataType.NORM, Joint.PM)

    # unit vectors shapes: (T, 3)

    unit_z = mid_mcp - origin  # vector from wrist to mid_mcp
    # divide by magnitude (becomes unit vector)
    unit_z /= np.linalg.norm(unit_z, axis=1, keepdims=True)

    vec_y = pinky_mcp - origin
    unit_x = np.cross(vec_y, unit_z)  # perpendicular
    unit_x /= np.linalg.norm(unit_x, axis=1, keepdims=True)

    unit_y = np.cross(unit_z, unit_x)

    # rotational matrix for linear transformation with time steps
    # (3, T, 3)
    R: np.ndarray = np.array([-unit_x, -unit_y, unit_z])

    R = R.transpose(1, 0, 2)  # (3, T, 3) -> (T, 3, 3)

    def change_of_basis(tip_idx: Joint, mcp_idx: Joint) -> np.ndarray:
        v = joint_data.get_positions(
            DataType.WORLD, tip_idx
        ) - joint_data.get_positions(DataType.WORLD, mcp_idx)  # (T, 3)

        return np.matmul(R, v[:, :, None]).squeeze(-1)

    # appendage vectors should be (T, 3)

    index = change_of_basis(Joint.IT, Joint.IM)
    middle = change_of_basis(Joint.MT, Joint.MM)
    ring = change_of_basis(Joint.RT, Joint.RM)
    thumb = change_of_basis(Joint.TT, Joint.TM)

    # concats 4 fingers' 3 vector components horizontally, retaining time
    result = np.concatenate([index, middle, ring, thumb], axis=1)  # (T, 12)

    return result


def process_deltas(data: np.ndarray) -> np.ndarray:
    deltas = np.diff(data, axis=0)  # deltas
    norm_deltas = normalize(deltas, deltas.max(), deltas.min(), 10, -10)
    round_data = norm_deltas.round(decimals=1)
    return round_data


def normalize(
    value,
    old_max: float,
    old_min: float,
    new_max: float,
    new_min: float,
):
    """
    Converts a number range to another range while maintaining ratio.
    """
    old_range = old_max - old_min
    new_range = new_max - new_min
    new_value = (((value - old_min) * new_range) / old_range) + new_min

    return new_value


def min_max_npy(directory_path):
    """
    Finds the overall minimum and maximum values across all .npy files in a given directory.

    Args:
        directory_path (str): The path to the directory containing the .npy files.

    Returns:
        tuple: A tuple containing (overall_min, overall_max).
    """
    overall_min = 0
    overall_max = 0
    found_npy_files = False

    for filename in os.listdir(directory_path):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory_path, filename)
            try:
                data = np.load(filepath)

                # Initialize overall_min and overall_max with the first file's min/max
                if not found_npy_files:
                    overall_min = np.min(data)
                    overall_max = np.max(data)
                    found_npy_files = True
                else:
                    overall_min = min(overall_min, np.min(data))
                    overall_max = max(overall_max, np.max(data))

            except Exception as e:
                print(f"Error loading or processing {filename}: {e}")

    return overall_min, overall_max


if __name__ == "__main__":
    # example usage:
    # Replace with the actual path to your directory
    directory = "/home/prabhune/projects/research/2026/eeg/data/"
    min_val, max_val = min_max_npy(directory)

    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")

    min_value = normalize(min_val, max_val, min_val, 1, -1)
    max_value = normalize(max_val, max_val, min_val, 1, -1)
    middle = normalize(-48.5, max_val, min_val, 1, -1)

    print(f"Normalized Minimum value: {min_value}")
    print(f"Normalized middle value: {middle}")
    print(f"Normalized Maximum value: {max_value}")
