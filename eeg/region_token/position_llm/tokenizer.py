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


class DeltaTokenizer(Tokenizer):
    """
    DeltaTokenizer converts a list of position "deltas"
    for a single joint (x, y, OR z) into tokens. This is just
    an example, since our full data is multidimensional
    (i.e. we have 21 joints with x, y, and z data for each
    => 63 columns per timestep). For the full data, we will have
    to devise a more complicated "quantization" scheme, which will
    take our continuous data and convert it into discrete tokens.

    E.g.:

    0: -10
    1: -9
    2: -8
    ...
    20: 10

    Going from delta values to tokens:
    [-8, -3, 4, 8, 2, 0, 5] => [2, 7, 14, 18, 12, 10, 15]
    """

    def __init__(self):
        self.mapping: list[float] = [i / 10.0 for i in range(-100, 110)]

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        tokens = []
        for delta in data:
            tokens.append(self.mapping.index(delta))

        return torch.tensor(tokens)

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        deltas = []
        for token in data:
            deltas.append(self.mapping[token])
        return torch.tensor(deltas, dtype=torch.float64)


class RegionTokenizer(Tokenizer):
    """
    RegionTokenizer uses a trained and scaled KMeans model
    on raw position values to produce 50 clusters.
    """

    def __init__(self, model_location: str):
        self.scaler: StandardScaler = joblib.load(
            f"{model_location}/kmeans_scaler.joblib"
        )
        self.kmeans: KMeans = joblib.load(f"{model_location}/kmeans.joblib")
        self.region_centers = self.kmeans.cluster_centers_

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        # normalize data
        scaled_data = self.scaler.transform(data.float().numpy())
        regions = self.kmeans.predict(scaled_data)

        return torch.tensor(regions)

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Returns the region center for each region token based on the KMeans.
        """
        centers = []
        for region in data:
            centers.append(self.region_centers[region])

        return torch.tensor(np.array(centers))
