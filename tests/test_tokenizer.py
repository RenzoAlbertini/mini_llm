import tempfile
import unittest
from pathlib import Path

from tokenizer.tokenizer import BPETokenizer, test_roundtrip


class TokenizerTest(unittest.TestCase):
    def test_roundtrip(self):
        tokenizer = BPETokenizer()
        text = "Ciao mini LLM! àèìòù"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertTrue(test_roundtrip(text, tokenizer))

    def test_save_load(self):
        tokenizer = BPETokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            tokenizer.save_model(path)
            loaded = BPETokenizer.load_model(path)
            self.assertEqual(loaded.decode(loaded.encode("test")), "test")


if __name__ == "__main__":
    unittest.main()
