import argparse
import json
import math
from pathlib import Path


def average_metrics(model, loader, device, max_batches):
    import torch

    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)
            losses.append(loss.item())
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    loss = sum(losses) / max(1, len(losses))
    return loss, correct / max(1, total)


def evaluate_loaded(label, model, loader, device, max_batches):
    loss, accuracy = average_metrics(model, loader, device, max_batches)
    return {
        "label": label,
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "accuracy": accuracy,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(description="Valuta loss media e perplexity sul validation set.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--quantized_checkpoint", default=None)
    parser.add_argument("--compare_dtypes", action="store_true")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--processed", default="data/processed/tokens.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--out_dir", default="data/eval")
    args = parser.parse_args()

    try:
        import torch

        from model.config import ModelConfig
        from tokenizer.tokenizer import BPETokenizer
        from training.dataset import create_dataloaders, load_or_tokenize
        from utils.helpers import get_device, load_checkpoint
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        return 1

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    config = ModelConfig.from_dict(checkpoint["config"])
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    config.vocab_size = tokenizer.vocab_size
    tokens = load_or_tokenize(args.data_dir, args.tokenizer, args.processed)
    _, val_loader = create_dataloaders(tokens, config.seq_len, args.batch_size, val_fraction=args.val_fraction)

    from inference.generate import load_model

    results = []
    normal = load_model(args.checkpoint, device, quantized=False)
    row = evaluate_loaded("fp32", normal, val_loader, device, args.max_batches)
    row["checkpoint"] = args.checkpoint
    results.append(row)

    if args.compare_dtypes and device.type == "cuda":
        fp16 = load_model(args.checkpoint, device, quantized=False).half()
        row = evaluate_loaded("fp16", fp16, val_loader, device, args.max_batches)
        row["checkpoint"] = args.checkpoint
        results.append(row)
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            bf16 = load_model(args.checkpoint, device, quantized=False).to(dtype=torch.bfloat16)
            row = evaluate_loaded("bf16", bf16, val_loader, device, args.max_batches)
            row["checkpoint"] = args.checkpoint
            results.append(row)

    if args.quantized_checkpoint:
        quantized = load_model(args.quantized_checkpoint, device, quantized=True)
        row = evaluate_loaded("8bit", quantized, val_loader, device, args.max_batches)
        row["checkpoint"] = args.quantized_checkpoint
        results.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluation.json"
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    print("model evaluation")
    print("================")
    for row in results:
        print(
            f"{row['label']:9s} loss={row['loss']:.4f} "
            f"ppl={row['perplexity']:.2f} acc={row['accuracy']:.4f}"
        )
    print(f"risultati salvati in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
