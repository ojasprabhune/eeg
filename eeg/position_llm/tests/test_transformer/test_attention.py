from transformer import MultiHeadAttention

import torch

import unittest

class TestMultiHeadAttention(unittest.TestCase):
    def test_split_heads(self):
        """Basic sanity test for splitting heads."""
        torch.manual_seed(42)
        mha = MultiHeadAttention(embedding_dim=32, num_heads=8, qk_length=32, value_length=32)
        x = torch.randn(32, 64, 8 * 32)
        Q, _, _ = x, x, x

        self.assertEqual(mha.split_heads(Q, mha.qk_length).size(), (32, 8, 64, 32))

    def test_split_heads_incorrect_shape(self):
        """Test that the input tensor must have the correct shape."""
        torch.manual_seed(42)
        mha = MultiHeadAttention(embedding_dim=32, num_heads=8, qk_length=32, value_length=32)
        x = torch.randn(32, 8 * 31, 64)
        Q, _, _ = x, x, x

        with self.assertRaises(AssertionError):
            mha.split_heads(Q, mha.qk_length)

    def test_combine_heads(self):
        """Basic sanity test for combining heads."""
        torch.manual_seed(42)
        mha = MultiHeadAttention(embedding_dim=32, num_heads=8, qk_length=32, value_length=32)
        x = torch.randn(32, 8, 64, 32)

        self.assertEqual(mha.combine_heads(x).size(), (32, 64, 8 * 32))

    def test_scaled_dot_product_attention(self):
        """Basic sanity test for scaled dot-product attention."""
        torch.manual_seed(42)
        mha = MultiHeadAttention(embedding_dim=32, num_heads=8, qk_length=32, value_length=32)
        Q = torch.randn(32, 8, 64, 32)
        K = torch.randn(32, 8, 64, 32)
        V = torch.randn(32, 8, 64, 32)

        self.assertEqual(mha.scaled_dot_product_attention(Q, K, V).size(), (32, 8, 64, 32))

    def test_forward(self):
        """Basic sanity test for the forward pass."""
        torch.manual_seed(42)
        mha = MultiHeadAttention(embedding_dim=32, num_heads=8, qk_length=32, value_length=32)

        # (B, T, C)
        x = torch.randn(32, 64, 32)
        Q, K, V = x, x, x

        self.assertEqual(mha(Q, K, V).size(), (32, 64, 32))
