import torch
import torch.nn as nn


class EEGRegressionModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        """
        Docstring for __init__
        
        :param self: Description
        :param input_dim: Description
        :type input_dim: int
        :param output_dim: Description
        :type output_dim: int
        """

        super().__init__()

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Docstring for forward

        :param self: Description
        :param x: Description
        :type x: torch.Tensor
        """

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x
