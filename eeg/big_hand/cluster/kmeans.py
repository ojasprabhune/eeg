import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from eeg.data_collection import JointData, Joint, DataType
from ..position_llm.utils import appendages


def load_data(data_file: str) -> np.ndarray:
    """
    Load all position data that we will use to train our KMeans model.
    In the future, when we have all the data, we should load all of
    it here and then train our model.
    """
    data = np.load(data_file)  # (T, 63)
    return data


def preprocess_data(
    data_file: str, show: bool = False
) -> tuple[np.ndarray, StandardScaler]:
    """
    Calculate appendage vectors over time and scale.
    """

    scaler: StandardScaler = StandardScaler()
    joint_data = JointData(data_file)

    app_data = appendages(joint_data)  # (T, 12)
    scaler_data: np.ndarray = scaler.fit_transform(app_data)  # (T, 12)

    print("Scalar data shape:", scaler_data.shape)

    return scaler_data, scaler


def kmeans(save_location: str, data_file: str) -> None:
    # find appendages for data and scaler

    data, scaler = preprocess_data(data_file)

    # init is method of initializing clusters. n_init means it runs
    # KMeans 20 times with different initial clusters. n_clusters
    # is the number of regions (number of region tokens) we will get
    kmeans = KMeans(n_clusters=50, init="k-means++", n_init=20).fit(data)

    # save scaler and KMeans model. need scaler in future because scaler
    # will process data properly
    scaler_filename = f"{save_location}\\kmeans_scaler.joblib"
    kmeans_filename = f"{save_location}\\kmeans.joblib"

    joblib.dump(scaler, scaler_filename)
    joblib.dump(kmeans, kmeans_filename)
