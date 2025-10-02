import numpy as np
import os


def process_deltas(data: np.ndarray) -> np.ndarray:
    deltas = np.diff(data, axis=0)  # deltas
    norm_deltas = normalize(deltas, deltas.max(), deltas.min(), 20, -20)
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
    directory = "/home/prabhune/projects/research/2026/eeg/data/"  # Replace with the actual path to your directory
    min_val, max_val = min_max_npy(directory)

    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")

    min_value = normalize(min_val, max_val, min_val, 1, -1)
    max_value = normalize(max_val, max_val, min_val, 1, -1)
    middle = normalize(-48.5, max_val, min_val, 1, -1)

    print(f"Normalized Minimum value: {min_value}")
    print(f"Normalized middle value: {middle}")
    print(f"Normalized Maximum value: {max_value}")
