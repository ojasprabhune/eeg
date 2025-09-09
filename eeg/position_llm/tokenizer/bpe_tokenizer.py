from .tokenizer import Tokenizer

import torch
from transformers import AutoTokenizer

class BPETokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the BPETokenizer class for French to English translation.

        Uses a pretrained BPE tokenizer to encode and decode text.
        """

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self.vocab = self.tokenizer.get_vocab()
    
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text, max_length=1024))
    
    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens.tolist())
