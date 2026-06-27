import unittest


try:
    import torch

    from model.config import ModelConfig
    from model.transformer import MiniTransformerLM
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch non installato")
class ModelForwardTest(unittest.TestCase):
    def test_forward(self):
        config = ModelConfig(vocab_size=300, seq_len=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)
        model = MiniTransformerLM(config)
        x = torch.randint(0, config.vocab_size, (2, config.seq_len))
        logits, loss = model(x, x)
        self.assertEqual(logits.shape, (2, config.seq_len, config.vocab_size))
        self.assertIsNotNone(loss)


if __name__ == "__main__":
    unittest.main()
