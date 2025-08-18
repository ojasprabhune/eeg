import numpy as np
import matplotlib.pyplot as plt


class JointData:
    def __init__(self, file_path: str):
        self.file_path = file_path # set file path
        self.data = np.load(file_path) # read file from file path


    # get function to obtain positions for a landmark
    def get_positions(self, joint_name, component: str = None):
        from eeg.data_collection import Joints
        # [:, ...] -> the :, means include all rows
        index = Joints[joint_name] # get the value from Enum
        start = index * 3  # one per component

        if component is None:
            positions = self.data[:, start : start + 3]
        elif component == "x":
            positions = self.data[:, start : start + 1]
        elif component == "y":
            positions = self.data[:, start + 1 : start + 2]
        elif component == "z":
            positions = self.data[:, start + 2 : start + 3]
        else:
            raise ValueError(f"Unknown component: {component}")
        return positions


    def get_dataset(self):
        return self.data


    # plot function for a joint 
    def plot_data(self, joint_name):


        def plot(x: list, y: list, ax: np.ndarray, xlabel: str, ylabel: str):
            ax.plot(x, y) # plot graph
            ax.set_xlabel(xlabel) # x label
            ax.set_ylabel(ylabel) # y label


        fig, ax = plt.subplots(3, 2, figsize=(20, 16))

        x_data = self.get_positions(joint_name, "x") 
        y_data = self.get_positions(joint_name, "y") 
        z_data = self.get_positions(joint_name, "z") 
        
        time = np.linspace(0, len(x_data), len(x_data))

        plot(time, x_data, ax[0, 0], "Time (frames)", "Expected relative X position")
        plot(time, y_data, ax[1, 0], "Time (frames)", "Expected relative Y position")
        plot(time, z_data, ax[2, 0], "Time (frames)", "Expected relative Z position")
        # axis=0 in order to unpack sublists and differentiate between single values
        plot(np.delete(time, -1), np.diff(x_data, axis=0), ax[0, 1], "Time (frames)", "Expected X delta")
        plot(np.delete(time, -1), np.diff(y_data, axis=0), ax[1, 1], "Time (frames)", "Expected Y delta")
        plot(np.delete(time, -1), np.diff(z_data, axis=0), ax[2, 1], "Time (frames)", "Expected Z delta")

        plt.show()