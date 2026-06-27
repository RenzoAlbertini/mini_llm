import argparse
from pathlib import Path

from config_manager import build_configs
from model.config import ModelConfig
from training.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning semplice di mini_llm su dataset custom.")
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--processed", default="data/processed/finetune_tokens.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints/finetune")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum_start_seq_len", type=int, default=64)
    parser.add_argument("--curriculum_step_size", type=int, default=100)
    parser.add_argument("--curriculum_increment", type=int, default=64)
    parser.add_argument("--data_source", action="append", default=[])
    parser.add_argument("--quantized_base", action="store_true", help="Carica base 8-bit dequantizzata prima del fine-tuning.")
    parser.add_argument("--save_quantized", action="store_true")
    parser.add_argument("--stats_path", default="data/logs/finetune_stats.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch

        from inference.generate import load_model
        from tokenizer.tokenizer import BPETokenizer
        from utils.helpers import get_device
        from utils.quantization import save_quantized_model
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        return 1

    device = get_device()
    base_model = load_model(args.base_checkpoint, device, quantized=args.quantized_base)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.checkpoint_dir) / "resume_from_base.pt"
    torch.save({"model": base_model.state_dict(), "config": base_model.config.to_dict(), "step": 0, "epoch": 0}, resume_path)
    args.resume = str(resume_path)

    tokenizer = BPETokenizer.load_model(args.tokenizer)
    config = base_model.config
    config.vocab_size = tokenizer.vocab_size
    final_path = train_model(args, config=config)

    if args.save_quantized:
        model = load_model(final_path, device, quantized=False)
        save_quantized_model(model, Path(args.checkpoint_dir) / "final_quantized.pt", config=model.config)
    print(f"fine-tuning completato: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
