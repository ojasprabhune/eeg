import torch
import numpy as np


class Tokenizer:
    def encode(self, data: torch.Tensor):
        pass

    def decode(self, data: torch.Tensor):
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
