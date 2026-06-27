import unittest


try:
    import torch

    from inference.generate import generate
    from model.config import ModelConfig
    from model.transformer import MiniTransformerLM
    from tokenizer.tokenizer import BPETokenizer
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch non installato")
class GenerateTest(unittest.TestCase):
    def test_generate_runs(self):
        tokenizer = BPETokenizer()
        config = ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=16, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        model = MiniTransformerLM(config)
        text = generate(model, tokenizer, "ciao", max_new_tokens=3, top_k=10, top_p=0.9, device=torch.device("cpu"))
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
