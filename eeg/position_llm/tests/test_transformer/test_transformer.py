from transformer import Transformer

import torch

import unittest

class TestTransformer(unittest.TestCase):
    def test_transformer(self):
        """Basic sanity test for the Transformer."""
        torch.manual_seed(42)
        transformer = Transformer(vocab_size=100,
                                  num_layers=6, 
                                  num_heads=8, 
                                  embedding_dim=32, 
                                  ffn_hidden_dim=64, 
                                  qk_length=32,
                                  value_length=32,
                                  max_length=500,
                                  dropout=0.1)
        src = torch.randint(0, 100, (32, 64))
        tgt = torch.randint(0, 100, (32, 64))

        self.assertEqual(transformer(src, tgt).size(), (32, 64, 100))
