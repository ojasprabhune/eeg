import torch


class Tokenizer:
    def encode(self, data: list[str]) -> torch.Tensor:
        pass

    def decode(self, data: torch.Tensor) -> list[str]:
        pass


class LanguageTokenizer(Tokenizer):
    """ """

    def __init__(self) -> None:
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}

        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", start=3):
            vocab[c] = i

        self.token_to_letter = {v: k for k, v in vocab.items()}
        self.letter_to_token = vocab

    def encode(self, data: list[str]) -> torch.Tensor:
        """
        Encodes a list of sentences of letters into a list of sequences of
        tokens.
        """
        encoded = []

        # add 2 because <SOS> and <EOS>
        self.max_seq_len = max(len(sentence) for sentence in data) + 2

        for sentence in data:
            tokens = [self.letter_to_token["<SOS>"]]  # start with <SOS>

            for letter in sentence:
                tokens.append(self.letter_to_token[letter])

            tokens.append(self.letter_to_token["<EOS>"])

            # pad until max sequence length
            tokens = tokens + [self.letter_to_token["<PAD>"]] * (
                self.max_seq_len - len(tokens)
            )
            encoded.append(tokens)

        return torch.tensor(encoded)

    def decode(self, data: torch.Tensor) -> list[str]:
        """
        Decodes a list of sequences of tokens into a list of sentences of
        letters.
        """
        decoded = []

        for sequence in data:
            letters = []

            for token in sequence:
                letter = self.token_to_letter[int(token.item())]

                # skip special tokens
                if letter in ("<SOS>", "<EOS>", "<PAD>"):
                    continue

                letters.append(letter)

            decoded.append("".join(letters))

        return decoded
