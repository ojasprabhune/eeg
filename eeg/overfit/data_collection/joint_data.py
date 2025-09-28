import numpy as np
import matplotlib.pyplot as plt

from .joints import Joint


class JointData:
    def __init__(self, file_path: str):
        self.file_path = file_path  # set file path
        self.data = np.load(file_path)  # read file from file path

    # get function to obtain positions for a landmark
    def get_positions(self, joint: str | Joint, component: str = None):
        # [:, ...] -> the :, means include all rows
        if isinstance(joint, str):
            joint = Joint.from_str(joint)
        index = joint.value  # get the value from Enum
        start = index * 3  # one per component

        if component is None:
            positions = self.data[:, start : start + 3]
        elif component == "x":
            positions = self.data[:, start]
        elif component == "y":
            positions = self.data[:, start + 1]
        elif component == "z":
            positions = self.data[:, start + 2]
        else:
            raise ValueError(f"Unknown component: {component}")
        return positions

    def get_dataset(self):
        return self.data

    # plot function for a joint
    def plot_data(self, joint: str | Joint):
        if isinstance(joint, str):
            joint = Joint.from_str(joint)

        def plot(x: list, y: list, ax: np.ndarray, xlabel: str, ylabel: str):
            ax.plot(x, y)  # plot graph
            ax.set_xlabel(xlabel)  # x label
            ax.set_ylabel(ylabel)  # y label

        fig, ax = plt.subplots(3, 2, figsize=(10, 8))

        x_data = self.get_positions(joint, "x")
        y_data = self.get_positions(joint, "y")
        z_data = self.get_positions(joint, "z")

        time = np.linspace(0, len(x_data), len(x_data))

        plot(time, x_data, ax[0, 0], "Time (frames)", "Expected relative X position")
        plot(time, y_data, ax[1, 0], "Time (frames)", "Expected relative Y position")
        plot(time, z_data, ax[2, 0], "Time (frames)", "Expected relative Z position")
        # axis=0 in order to unpack sublists and differentiate between single values
        plot(
            np.delete(time, -1),
            np.diff(x_data, axis=0),
            ax[0, 1],
            "Time (frames)",
            "Expected X delta",
        )
        plot(
            np.delete(time, -1),
            np.diff(y_data, axis=0),
            ax[1, 1],
            "Time (frames)",
            "Expected Y delta",
        )
        plot(
            np.delete(time, -1),
            np.diff(z_data, axis=0),
            ax[2, 1],
            "Time (frames)",
            "Expected Z delta",
        )

        plt.show()

    @staticmethod
    def plot_model_traces(
        expected_data: np.ndarray, predicted_data: np.ndarray, joint: Joint
    ):
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))

        x_data = expected_data[:, joint.value * 3]
        y_data = expected_data[:, joint.value * 3 + 1]
        z_data = expected_data[:, joint.value * 3 + 2]

        pred_x_data = predicted_data[:, joint.value * 3]
        pred_y_data = predicted_data[:, joint.value * 3 + 1]
        pred_z_data = predicted_data[:, joint.value * 3 + 2]

        time = np.linspace(0, len(x_data), len(x_data))

        ax[0].plot(time, x_data, label="Expected")
        ax[0].plot(time, pred_x_data, label="Predicted")
        ax[0].set_xlabel("Time (frames)")
        ax[0].set_ylabel("X position")
        ax[0].legend()
        ax[1].plot(time, y_data, label="Expected")
        ax[1].plot(time, pred_y_data, label="Predicted")
        ax[1].set_xlabel("Time (frames)")
        ax[1].set_ylabel("Y position")
        ax[1].legend()
        ax[2].plot(time, z_data, label="Expected")
        ax[2].plot(time, pred_z_data, label="Predicted")
        ax[2].set_xlabel("Time (frames)")
        ax[2].set_ylabel("Z position")
        ax[2].legend()
        ax[1].plot(time, y_data)
        ax[1].set_xlabel("Time (frames)")
        ax[1].set_ylabel("Y position")

        ax[2].plot(time, z_data)
        ax[2].set_xlabel("Time (frames)")
        ax[2].set_ylabel("Z position")

        plt.show()

