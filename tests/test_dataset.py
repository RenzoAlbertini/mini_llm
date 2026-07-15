import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from tokenizer.tokenizer import BPETokenizer
    from training.dataset import create_dataloaders, load_or_tokenize
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch non installato")
class DatasetTest(unittest.TestCase):
    def test_train_validation_windows_do_not_overlap(self):
        tokens = torch.arange(1000)
        train_loader, val_loader = create_dataloaders(
            tokens, seq_len=32, batch_size=8, val_fraction=0.2, stride=1
        )
        train_tokens = train_loader.dataset.tokens
        val_tokens = val_loader.dataset.tokens
        self.assertLess(int(train_tokens[-1]), int(val_tokens[0]))
        self.assertGreaterEqual(int(val_tokens[0]) - int(train_tokens[-1]), 32)

    def test_small_validation_fraction_still_has_a_complete_window(self):
        tokens = torch.arange(1000)
        train_loader, val_loader = create_dataloaders(
            tokens, seq_len=64, batch_size=4, val_fraction=0.05, stride=1
        )
        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)

    def test_token_cache_is_invalidated_when_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.txt"
            tokenizer_path = root / "tokenizer.json"
            cache = root / "tokens.pt"
            BPETokenizer().save_model(tokenizer_path)
            source.write_text("prima versione", encoding="utf-8")
            first = load_or_tokenize(source, tokenizer_path, cache)
            source.write_text("seconda versione piu lunga", encoding="utf-8")
            second = load_or_tokenize(source, tokenizer_path, cache)
            self.assertNotEqual(first.tolist(), second.tolist())


if __name__ == "__main__":
    unittest.main()
