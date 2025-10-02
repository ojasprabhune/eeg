import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from eeg.region_token.data_collection.utils import normalize, process_deltas
from eeg.region_token.position_llm import DeltaTokenizer


def load_data(data_file: str) -> np.ndarray:
    """
    Load all position data that we will use to train our KMeans model.
    In the future, when we have all the data, we should load all of
    it here and then train our model.
    """
    data = np.load(data_file)  # (T, 63)
    return data


def preprocess_data(
    data: np.ndarray, show: bool = False
) -> tuple[np.ndarray, StandardScaler]:
    """
    Preprocess our data using normalization and other techniques so that
    KMeans will train correctly.

    1. Scale data so mean is 0 and mean deviation is 1.
    2. Find deltas.
    3. Normalize from minimum and maximum to -10 and 10 while keeping ratio.
    """

    scaler: StandardScaler = StandardScaler()
    delta_tokenizer = DeltaTokenizer()

    scaler_data: np.ndarray = scaler.fit_transform(data)  # still (T, 63)
    round_data = process_deltas(scaler_data)
    delta_tokens = delta_tokenizer.encode(round_data)
    # encode deltas into delta tokens ^

    print("Delta tokens:", delta_tokens.shape)

    return delta_tokens, scaler


def kmeans(save_location: str, data_file: str) -> None:
    data = load_data(data_file)  # get file
    data, scaler = preprocess_data(data)  # preprocess it and data and scaler

    # init is method of initializing clusters. n_init means it runs
    # KMeans 20 times with different initial clusters. n_clusters
    # is the number of regions (number of region tokens) we will get
    kmeans = KMeans(n_clusters=50, init="k-means++", n_init=20).fit(data)

    # save scaler and KMeans model. need scaler in future because scaler
    # will process data properly
    scaler_filename = f"{save_location}/kmeans_scaler.joblib"
    kmeans_filename = f"{save_location}/kmeans.joblib"

    joblib.dump(scaler, scaler_filename)
    joblib.dump(kmeans, kmeans_filename)
