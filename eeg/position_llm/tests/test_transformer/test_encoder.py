from transformer import EncoderLayer, Encoder

import torch

import unittest

class TestEncoder(unittest.TestCase):
    def test_encoder_layer(self):
        """Basic sanity test for the EncoderLayer."""
        torch.manual_seed(42)
        encoder_layer = EncoderLayer(num_heads=8, 
                                     embedding_dim=32, 
                                     ffn_hidden_dim=64, 
                                     qk_length=32, 
                                     value_length=32, 
                                     dropout=0.1)
        x = torch.randn(32, 64, 32)
        Q, _, _ = x, x, x

        self.assertEqual(encoder_layer(Q).size(), (32, 64, 32))

    def test_encoder(self):
        """Basic sanity test for the Encoder."""
        torch.manual_seed(42)
        encoder = Encoder(vocab_size=100, 
                          num_layers=6, 
                          num_heads=8, 
                          embedding_dim=32, 
                          ffn_hidden_dim=64, 
                          qk_length=32,
                          value_length=32,
                          max_length=100,
                          dropout=0.1)
        x = torch.randint(0, 100, (32, 64))
        self.assertEqual(encoder(x).size(), (32, 64, 32))
