import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Valida checkpoint, tokenizer e generazione.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--min_tokens", type=int, default=8)
    args = parser.parse_args()

    try:
        import torch

        from config_manager import validate_model_config
        from inference.generate import generate, load_model
        from tokenizer.tokenizer import BPETokenizer
        from utils.helpers import get_device
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        return 1

    if not Path(args.checkpoint).exists():
        print(f"FAIL: checkpoint non trovato: {args.checkpoint}")
        return 1
    if not Path(args.tokenizer).exists():
        print(f"FAIL: tokenizer non trovato: {args.tokenizer}")
        return 1

    device = get_device()
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    model = load_model(args.checkpoint, device, quantized=args.quantized)
    config = model.config
    validate_model_config(config)

    print("model validation")
    print("================")
    print(f"vocab tokenizer: {tokenizer.vocab_size}")
    print(f"vocab model:     {config.vocab_size}")
    print(f"seq_len:         {config.seq_len}")
    print(f"layers:          {config.n_layers}")
    print(f"heads:           {config.n_heads}")
    print(f"d_model:         {config.d_model}")

    if tokenizer.vocab_size != config.vocab_size:
        print("FAIL: tokenizer vocab_size diverso dal modello")
        return 1

    for name, tensor in model.state_dict().items():
        if not torch.isfinite(tensor.float()).all():
            print(f"FAIL: tensore non finito: {name}")
            return 1
    print("pesi: OK")

    text = generate(model, tokenizer, "python is", max_new_tokens=args.min_tokens, device=device)
    generated = tokenizer.encode(text)
    if len(generated) < args.min_tokens:
        print("FAIL: generazione troppo corta")
        return 1
    print(f"generazione: OK ({len(generated)} token totali)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
