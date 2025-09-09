from typing import List, Dict, Tuple, Optional

import torch


class Tokenizer:
    """
    Tokenizer class that tokenizes the input text.

    You will need TWO functions in this class. You
    do not need to have a body here, rather the body
    will be added in the CharacterTokenizer subclass.

    This class is simply a template that could be
    used in the future for other tokenizers.
    """

    def encode(self, input: str):
        pass

    def decode(self, input: torch.Tensor):
        pass
