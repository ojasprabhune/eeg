import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Tokenizer:
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        pass

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        pass


class RegionTokenizer(Tokenizer):
    """
    RegionTokenizer uses a trained and scaled KMeans model
    on appendage vectors to produce 50 clusters. Each cluster
    is a "region" in the 12-dimensional space of vector components.
    Each region is represented by a token (0-49). The encode
    function converts appendage vector components into region tokens,
    while the decode function converts region tokens back into
    the center position of that region.
    """

    def __init__(self, model_location: str):
        self.scaler: StandardScaler = joblib.load(
            f"{model_location}/kmeans_scaler.joblib"
        )
        self.kmeans: KMeans = joblib.load(f"{model_location}/kmeans.joblib")
        self.region_centers = self.kmeans.cluster_centers_

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Scales input data, predicts region clusters, and returns a torch
        Tensor of size (T,).
        """
        # normalize data
        scaled_data = self.scaler.transform(data)
        regions = self.kmeans.predict(scaled_data)

        return torch.tensor(regions)  # (T,)

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Returns the region center for each region token based on the KMeans.
        """
        centers = []
        for region in data:
            centers.append(self.region_centers[region])

        return torch.tensor(np.array(centers))


class DeltaTokenizer(Tokenizer):
    """
    DeltaTokenizer converts sequences of position deltas (T, 63)
    into tokens. Each delta value is mapped to a token based on
    a predefined mapping from -10.0 to 10.0 in increments of 0.1.
    Each time step contains 63 delta values, resulting in a
    sequence of tokenized time steps.

    E.g.

    0: -10.0
    1: -9.9
    2: -9.8
    ...
    200: 10.0

    Going from delta values to tokens:
    [[-8.0, -3.0, 4.0, ..., 0.0], ...] => [[20, 71, 140, ..., 100
    """

    def __init__(self):
        self.mapping: list[float] = [i / 10.0 for i in range(-100, 110)]
        # self.mapping: list[float] = [i / 10.0 for i in range(-200, 201)]

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        # data is shape (T, 63)
        tokens = []
        for time_step in data:  # time_step is shape (63)
            time_steps = []
            for delta in time_step:  # 63 deltas in each time step
                # append 63 deltas to each time step
                time_steps.append(self.mapping.index(delta))
            tokens.append(time_steps)

        return torch.tensor(tokens)

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        deltas = []
        for time_step in data:
            time_steps = []
            for token in time_step:
                time_steps.append(self.mapping[token])
            deltas.append(time_steps)

        return torch.tensor(deltas, dtype=torch.float64)
