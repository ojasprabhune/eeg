from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = "aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

    # encode string to tokens
    def encode(self, input: str) -> torch.Tensor:
        tokens = []
        for char in list(input):  # list(input) is a list of char
            tokens.append(ord(char))  # ord returns Unicode code point
        return torch.tensor(tokens)  # convert to tensor

    # decode tokens to string
    def decode(self, input: torch.Tensor) -> str:
        output = ""
        for token in input:  # input is a list of tokens
            output += chr(token)  # chr() is opposite of ord() (string concatenation)
        return output.lower()  # capitalization doesn't matter

    # encoding/decoding test: python -m unittest tests/test_tokenizer/test_character_tokenizer.py
