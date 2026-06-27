import argparse


def main():
    parser = argparse.ArgumentParser(description="Profiling GPU di una forward/backward mini_llm.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--out_dir", default="data/profiling")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    try:
        import torch

        from inference.generate import load_model
        from tokenizer.tokenizer import BPETokenizer
        from utils.helpers import get_device, gpu_profile
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        return 1

    device = get_device()
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    model = load_model(args.checkpoint, device)
    seq_len = min(32, model.config.seq_len)
    x = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
    y = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
    report = gpu_profile(model, (x, y), out_dir=args.out_dir, steps=args.steps)
    print(f"profiling salvato in {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
