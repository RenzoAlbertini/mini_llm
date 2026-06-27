import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Controlla tokenizer, dataset e modello prima del training.")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--processed", default="data/processed/tokens.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seq_len", type=int, default=None)
    args = parser.parse_args()

    try:
        import torch

        from model.config import default_small
        from model.transformer import MiniTransformerLM
        from tokenizer.tokenizer import BPETokenizer
        from training.dataset import create_dataloaders, load_or_tokenize
        from utils.helpers import count_parameters, get_device
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        print("Installa PyTorch con: pip install -r requirements.txt")
        return 1

    if not Path(args.tokenizer).exists():
        print(f"FAIL: tokenizer non trovato: {args.tokenizer}")
        print("Crea il tokenizer con: python -m tokenizer.build_tokenizer --data_dir data/raw --out tokenizer/tokenizer.json")
        return 1

    tokenizer = BPETokenizer.load_model(args.tokenizer)
    config = default_small()
    config.vocab_size = tokenizer.vocab_size
    if args.seq_len is not None:
        config.seq_len = args.seq_len

    tokens = load_or_tokenize(args.data_dir, args.tokenizer, args.processed)
    train_loader, val_loader = create_dataloaders(
        tokens,
        config.seq_len,
        args.batch_size,
        val_fraction=args.val_fraction,
    )
    model = MiniTransformerLM(config)
    device = get_device()

    print("pre-training check")
    print("==================")
    print(f"device: {device}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"seq_len: {config.seq_len}")
    print(f"tokens: {len(tokens)}")
    print(f"train_batches: {len(train_loader)}")
    print(f"val_batches: {len(val_loader)}")
    print(f"parameters: {count_parameters(model):,}")

    x, y = next(iter(train_loader))
    compatible = x.shape[1] == config.seq_len and y.shape[1] == config.seq_len
    print(f"batch_x: {tuple(x.shape)}")
    print(f"batch_y: {tuple(y.shape)}")
    print(f"config_compatible: {'OK' if compatible else 'FAIL'}")

    with torch.no_grad():
        logits, loss = model(x, y)
    print(f"forward_logits: {tuple(logits.shape)}")
    print(f"forward_loss: {loss.item():.4f}")
    print("summary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
