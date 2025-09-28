import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        qk_length: int,
        value_length: int,
    ):
        """
        The Multi-Head Attention layer will take in Q, K, and V
        matrices and will output an attention matrix of shape <TODO>.

        First, Q, K, and V should be projected to have
        a shape of (B, T, C) where C = num_heads * qk_length
        (OR value_length). You are then expected to split
        the C dimension into num_heads different heads, each
        with shape (B, T, vec_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """

        super().__init__()

        self.num_heads = num_heads
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # (hint: number of Linear layers needed != 3)

        self.Wq = nn.Linear(embedding_dim, num_heads * qk_length)
        self.Wk = nn.Linear(embedding_dim, num_heads * qk_length)
        self.Wv = nn.Linear(embedding_dim, num_heads * value_length)
        self.Wo = nn.Linear(num_heads * value_length, embedding_dim)

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, vec_length).

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * vec_length
            vec_length: int, the length of the query/key/value vectors

        Returns:
            torch.Tensor of shape (B, num_heads, T, vec_length)
        """

        B, T, C = x.size()
        # make sure that input tensor has correct shape
        assert C // self.num_heads == vec_length
        x = x.view(
            B, T, self.num_heads, vec_length
        )  # first split C into two dimensions
        out = torch.transpose(x, 1, 2)  # transpose T and self.num_heads

        return out

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the num_heads different heads into a single tensor.
        Hint: check out the `contiguous` method in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, num_heads, T, vec_length)

        Returns:
            torch.Tensor of shape (B, T, num_heads * vec_length)
        """

        B, num_heads, T, vec_length = x.size()

        x = torch.transpose(x, 1, 2)
        # contiguous flattens it then view does more stuff
        out = x.contiguous().view(B, T, num_heads * vec_length)

        return out

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the scaled dot-product attention given Q, K, and V.

        Args:
            Q: torch.Tensor of shape (B, num_heads, T, qk_length)
            K: torch.Tensor of shape (B, num_heads, T, qk_length)
            V: torch.Tensor of shape (B, num_heads, T, value_length)
            mask: Optional torch.Tensor of shape (B, T, T) or None
        """

        # softmax((Q * Kt)/sqrt(qk)) * V

        # rescales elements in Tensor so they lie in range [0, 1] and sum to 1
        Kt = torch.transpose(K, 2, 3)  # transpose T and qk_length

        # transpose of K
        lookup = torch.matmul(Q, Kt)  # check similarity between queries and keys
        scaled_lookup = lookup / math.sqrt(self.qk_length)  # compensate for varience

        if mask is not None:
            # want to change values of scaled_lookup where mask is 0
            # masked == 0 creates new tensor of true and false
            scaled_lookup.masked_fill(mask == 0, float("-inf"))

        # similarities summed to 1 for last dimension
        attention = F.softmax(scaled_lookup, dim=-1)

        # new embeddings with relevant value from other embeddings
        out = torch.matmul(attention, V)

        return out

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Multi-Head Attention layer.

        Args:
            Q: torch.Tensor of shape (B, T, C)
            K: torch.Tensor of shape (B, T, C)
            V: torch.Tensor of shape (B, T, C)
            mask: Optional torch.Tensor of shape (B, T, T) or None

        Returns:
            torch.Tensor of shape (B, T, C)
        """

        # Q: (5, 4)
        # T = 5, C = 4

        # Wq = (4, 16) Q * W
        # 16 = 4 * 4 = num_heads * qk_length
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Q: (5, 16)

        Q = self.split_heads(Q, self.qk_length)

        # if Q.view(4, 5, 4)
        # 1. number order is same, but starting rows at different places
        # 2. first must split into (5, 4, 4)
        # 3. then transpose first 2 dimensions
        # Q: (4, 5, 4)
        # num_heads, T, qk_length
        # need to preserve T: time dim

        K = self.split_heads(K, self.qk_length)
        V = self.split_heads(V, self.value_length)

        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        attention = self.combine_heads(attention)
        out = self.Wo(attention)

        return out


class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        The Feed-Forward Neural Network layer will take in
        an input tensor of shape (B, T, C) and will output
        a tensor of the same shape.

        The FFNN will have two linear layers, with a ReLU
        activation function in between.

        Args:
            hidden_dim: int, the size of the hidden layer
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Define any layers you'll need in the forward pass
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
