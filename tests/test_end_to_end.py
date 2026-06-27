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
class EndToEndTest(unittest.TestCase):
    def test_forward_and_generation(self):
        tokenizer = BPETokenizer()
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            seq_len=16,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
        )
        model = MiniTransformerLM(config)
        batch = torch.randint(0, config.vocab_size, (2, config.seq_len))
        logits, loss = model(batch, batch)
        self.assertEqual(logits.shape, (2, config.seq_len, config.vocab_size))
        self.assertIsNotNone(loss)

        text = generate(
            model,
            tokenizer,
            "test",
            max_new_tokens=4,
            temperature=1.0,
            top_k=20,
            top_p=0.95,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
