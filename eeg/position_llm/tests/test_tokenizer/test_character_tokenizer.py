import unittest
from tokenizer.character_tokenizer import CharacterTokenizer

class TestCharacterTokenizer(unittest.TestCase):
    def test_e2e(self):
        """Test basic end-to-end example."""
        tokenizer = CharacterTokenizer()
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "hello")

    def test_empty_input(self):
        """Test empty input."""
        tokenizer = CharacterTokenizer()
        encoded = tokenizer.encode("")
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "")

    def test_french_input(self):
        """Test example from French-English dataset."""
        tokenizer = CharacterTokenizer()
        text = ("Souvent considérée comme la plus ancienne des sciences, "
                 "elle découle de notre étonnement et de nos q...")
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text.lower())

    def test_english_input(self):
        """Test example from English-French dataset."""
        tokenizer = CharacterTokenizer()
        text = ("The white light spectrum Codes in the light"
                " The electromagnetic spectrum Emission spectra Absorption...")
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text.lower())

if __name__ == '__main__':
    unittest.main()
