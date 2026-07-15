import unittest


try:
    import torch

    from inference.generate import apply_repetition_penalty, generate
    from model.config import ModelConfig
    from model.transformer import MiniTransformerLM
    from tokenizer.tokenizer import BPETokenizer
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch non installato")
class GenerateTest(unittest.TestCase):
    def make_model(self):
        tokenizer = BPETokenizer()
        config = ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=16, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        return tokenizer, MiniTransformerLM(config)

    def test_generate_runs(self):
        tokenizer, model = self.make_model()
        text = generate(model, tokenizer, "ciao", max_new_tokens=3, top_k=10, top_p=0.9, device=torch.device("cpu"))
        self.assertIsInstance(text, str)
        self.assertLessEqual(len(tokenizer.encode(text)), 3)

    def test_greedy_generation_is_deterministic(self):
        tokenizer, model = self.make_model()
        first = generate(model, tokenizer, "ciao", max_new_tokens=5, seed=1, device=torch.device("cpu"))
        second = generate(model, tokenizer, "ciao", max_new_tokens=5, seed=999, device=torch.device("cpu"))
        self.assertEqual(first, second)

    def test_seeded_sampling_is_reproducible(self):
        tokenizer, model = self.make_model()
        first = generate(model, tokenizer, "ciao", max_new_tokens=5, do_sample=True, seed=7, device=torch.device("cpu"))
        second = generate(model, tokenizer, "ciao", max_new_tokens=5, do_sample=True, seed=7, device=torch.device("cpu"))
        self.assertEqual(first, second)

    def test_repetition_penalty_is_sign_aware(self):
        logits = torch.tensor([[2.0, -2.0, 1.0]])
        penalized = apply_repetition_penalty(logits.clone(), [0, 1], penalty=2.0)
        self.assertEqual(float(penalized[0, 0]), 1.0)
        self.assertEqual(float(penalized[0, 1]), -4.0)
        self.assertEqual(float(penalized[0, 2]), 1.0)

    def test_invalid_sampling_parameters_fail_clearly(self):
        tokenizer, model = self.make_model()
        with self.assertRaisesRegex(ValueError, "top_p"):
            generate(model, tokenizer, "ciao", max_new_tokens=2, top_p=0, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
