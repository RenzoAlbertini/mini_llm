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

    def test_causal_mask_blocks_future_information(self):
        torch.manual_seed(0)
        config = ModelConfig(
            vocab_size=300, seq_len=8, d_model=32, n_layers=2,
            n_heads=4, d_ff=64, dropout=0.0,
        )
        model = MiniTransformerLM(config).eval()
        original = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
        changed = original.clone()
        changed[0, 5:] = torch.tensor([20, 21, 22])
        with torch.no_grad():
            logits_a, _ = model(original)
            logits_b, _ = model(changed)
        self.assertTrue(torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-6))

    def test_input_output_embeddings_are_tied(self):
        config = ModelConfig(vocab_size=300, seq_len=8, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        model = MiniTransformerLM(config)
        self.assertEqual(model.token_embedding.weight.data_ptr(), model.lm_head.weight.data_ptr())


if __name__ == "__main__":
    unittest.main()
