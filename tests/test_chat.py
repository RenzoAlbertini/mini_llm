import unittest
from unittest.mock import Mock, patch


try:
    import torch

    from chat.server import fallback_response, generate_chat_candidate, public_checkpoint_path
    from tokenizer.tokenizer import BPETokenizer
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/FastAPI non installati")
class ChatModeTest(unittest.TestCase):
    def test_every_chat_candidate_loads_and_runs_the_model(self):
        runtime = Mock()
        runtime.tokenizer = BPETokenizer()
        model = Mock()
        model.config.seq_len = 32
        runtime.load.return_value = (model, torch.device("cpu"))

        with patch("chat.server.generate", return_value="risposta prodotta dal modello") as mocked_generate:
            response = generate_chat_candidate(
                runtime,
                "checkpoint.pt",
                "User: ciao\nAssistant:",
                20,
                0.5,
                0.9,
                20,
            )

        self.assertEqual(response, "risposta prodotta dal modello")
        runtime.load.assert_called_once_with("checkpoint.pt")
        mocked_generate.assert_called_once()

    def test_coherent_model_output_is_not_replaced_by_template(self):
        candidate = "Questa risposta arriva dal modello locale ed e sufficientemente completa."
        self.assertEqual(fallback_response("ciao", candidate), candidate)

    def test_public_checkpoint_path_does_not_expose_parent_directories(self):
        path = public_checkpoint_path("models/checkpoints/best.pt")
        self.assertEqual(path, "models/checkpoints/best.pt")
        self.assertFalse(path.startswith("/"))


if __name__ == "__main__":
    unittest.main()
