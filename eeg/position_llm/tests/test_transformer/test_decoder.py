from transformer import DecoderLayer, Decoder

import torch

import unittest

class TestDecoder(unittest.TestCase):
    def test_decoder_layer(self):
        """Basic sanity test for the DecoderLayer."""
        torch.manual_seed(42)
        decoder_layer = DecoderLayer(num_heads=8, 
                                     embedding_dim=32, 
                                     ffn_hidden_dim=64, 
                                     qk_length=32, 
                                     value_length=32, 
                                     dropout=0.1)
        x = torch.randn(32, 64, 32)
        enc_x = torch.randn(32, 64, 32)
        mask = torch.ones(32, 8, 64, 64)
        Q, _, _ = x, x, x

        self.assertEqual(decoder_layer(Q, enc_x, mask).size(), (32, 64, 32))

    def test_decoder(self):
        """Basic sanity test for the Decoder."""
        torch.manual_seed(42)
        decoder = Decoder(vocab_size=100, 
                          num_layers=6, 
                          num_heads=8, 
                          embedding_dim=32, 
                          ffn_hidden_dim=64, 
                          qk_length=32,
                          value_length=32,
                          max_length=500,
                          dropout=0.1)
        tgt = torch.randint(0, 100, (32, 64))
        enc_x = torch.randn(32, 64, 32)

        self.assertEqual(decoder(tgt, enc_x).size(), (32, 64, 100))
