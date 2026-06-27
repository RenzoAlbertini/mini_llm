import argparse
import json
import random
import string
import time
from pathlib import Path


def random_prompt():
    words = ["python", "tensor", "model", "training", "data", "token", "attention", "loss"]
    return " ".join(random.choice(words) for _ in range(random.randint(2, 8)))


def main():
    parser = argparse.ArgumentParser(description="Stress test generazione mini_llm.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--prompts", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--out_dir", default="data/stress")
    args = parser.parse_args()

    try:
        from inference.generate import generate, load_model
        from tokenizer.tokenizer import BPETokenizer
        from utils.helpers import get_device
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        return 1

    device = get_device()
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    model = load_model(args.checkpoint, device, quantized=args.quantized)
    rows = []
    failures = 0
    started = time.perf_counter()
    for i in range(args.prompts):
        prompt = random_prompt()
        t0 = time.perf_counter()
        try:
            text = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, device=device)
            elapsed = time.perf_counter() - t0
            tokens = max(1, len(tokenizer.encode(text)) - len(tokenizer.encode(prompt, add_bos=True)))
            rows.append({"prompt": prompt, "seconds": elapsed, "tokens": tokens, "tokens_per_second": tokens / elapsed})
        except Exception as exc:
            failures += 1
            rows.append({"prompt": prompt, "error": f"{type(exc).__name__}: {exc}"})
    total = time.perf_counter() - started
    ok_rows = [r for r in rows if "tokens_per_second" in r]
    avg_tps = sum(r["tokens_per_second"] for r in ok_rows) / max(1, len(ok_rows))
    avg_latency = sum(r["seconds"] / max(1, r["tokens"]) for r in ok_rows) / max(1, len(ok_rows))
    report = {
        "prompts": args.prompts,
        "failures": failures,
        "seconds": total,
        "avg_tokens_per_second": avg_tps,
        "avg_latency_per_token_ms": avg_latency * 1000.0,
        "rows": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stress_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"stress test: failures={failures} avg_tps={avg_tps:.2f} report={out_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
