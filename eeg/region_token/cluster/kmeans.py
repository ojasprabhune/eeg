import joblib

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data() -> np.ndarray:
    """
    Load all position data that we will use to train our KMeans model.
    In the future, when we have all the data, we should load all of
    it here and then train our model.
    """
    data = np.load("data/open_fist_front.npy")
    return data


def preprocess_data(data: np.ndarray, show=False) -> tuple[np.ndarray, StandardScaler]:
    """
    Preprocess our data using normalization and other techniques so that
    KMeans will train correctly.
    """

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    if show:
        plt.hist(
            scaled_data[:, 24],
            bins=40,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            density=True,
        )
        plt.show()

    return scaled_data, scaler


def kmeans(save_location: str):
    data = load_data()
    data, scaler = preprocess_data(data)

    # init is method of initializing clusters. n_init means it runs KMeans 20 times
    # with different initial clusters.
    # n_clusters is the number of regions (number of region tokens) we will get
    kmeans = KMeans(n_clusters=50, init="k-means++", n_init=20).fit(data)

    scaler_filename = f"{save_location}/kmeans_scaler.joblib"
    kmeans_filename = f"{save_location}/kmeans.joblib"

    joblib.dump(scaler, scaler_filename)
    joblib.dump(kmeans, kmeans_filename)
