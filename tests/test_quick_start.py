import os
import tempfile
import unittest
from argparse import Namespace

from run_training import apply_mode, prepare_demo_assets


class QuickStartTest(unittest.TestCase):
    def test_debug_mode_turns_unlimited_steps_into_short_run(self):
        args = Namespace(
            mode="debug",
            batch_size=8,
            max_steps=0,
            eval_every=100,
            log_every=10,
            warmup_steps=None,
            seq_len=None,
        )
        apply_mode(args)
        self.assertEqual(args.max_steps, 20)

    def test_demo_assets_are_isolated_and_configuration_is_tiny(self):
        args = Namespace(
            data_dir="data/raw",
            processed="data/processed/real_tokens.pt",
            checkpoint_dir="models/checkpoints",
            stats_path="data/logs/training_stats.csv",
            plots_dir="data/plots",
            batch_size=8,
            epochs=3,
            max_steps=0,
            eval_every=100,
            log_every=10,
            warmup_steps=None,
            seq_len=None,
            d_model=None,
            n_layers=None,
            n_heads=None,
            d_ff=None,
            tokenizer="tokenizer/tokenizer.json",
            vocab_size=None,
        )
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                prepare_demo_assets(args)
                self.assertTrue(os.path.isfile(args.data_dir))
                self.assertTrue(os.path.isfile(args.tokenizer))
        finally:
            os.chdir(original_cwd)

        self.assertEqual(args.max_steps, 20)
        self.assertEqual(args.checkpoint_dir, "models/checkpoints/demo")
        self.assertEqual(args.seq_len, 64)
        self.assertEqual(args.d_model, 64)
        self.assertLessEqual(args.vocab_size, 512)


if __name__ == "__main__":
    unittest.main()
