import argparse
import csv
import json
import time
from pathlib import Path


def benchmark_mode(mode, args, tokenizer, device):
    import torch

    from inference.generate import generate, load_model

    quantized = mode == "8bit"
    checkpoint = args.quantized_checkpoint if quantized else args.checkpoint
    if quantized and not checkpoint:
        return {"mode": mode, "status": "SKIP", "reason": "checkpoint quantizzato non fornito"}
    if checkpoint is None or not Path(checkpoint).exists():
        return {"mode": mode, "status": "SKIP", "reason": f"checkpoint non trovato: {checkpoint}"}

    if mode in {"fp16", "bf16"} and device.type != "cuda":
        return {"mode": mode, "status": "SKIP", "reason": "richiede CUDA"}

    model = load_model(checkpoint, device, quantized=quantized)
    if mode == "fp16":
        model = model.half()
    elif mode == "bf16":
        model = model.to(dtype=torch.bfloat16)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated_tokens = max(1, len(tokenizer.encode(text)) - len(tokenizer.encode(args.prompt, add_bos=True)))
    return {
        "mode": mode,
        "status": "OK",
        "checkpoint": checkpoint,
        "seconds": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / elapsed,
        "latency_per_token_ms": elapsed / generated_tokens * 1000.0,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark semplice di inferenza mini_llm.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--quantized_checkpoint", default=None)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", default="python is")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--out_dir", default="data/benchmarks")
    args = parser.parse_args()

    try:
        from tokenizer.tokenizer import BPETokenizer
        from utils.helpers import get_device
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        return 1

    tokenizer = BPETokenizer.load_model(args.tokenizer)
    device = get_device()
    modes = ["fp32", "fp16", "bf16", "8bit"]
    results = []
    for mode in modes:
        try:
            result = benchmark_mode(mode, args, tokenizer, device)
        except Exception as exc:
            result = {"mode": mode, "status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
        results.append(result)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "inference_benchmark.json"
    csv_path = out_dir / "inference_benchmark.csv"
    json_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["mode", "status", "seconds", "generated_tokens", "tokens_per_second", "latency_per_token_ms", "device", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    print("inference benchmark")
    print("===================")
    for row in results:
        if row["status"] == "OK":
            print(
                f"{row['mode']:5s} OK | {row['tokens_per_second']:.2f} tok/s | "
                f"{row['latency_per_token_ms']:.2f} ms/token"
            )
        else:
            print(f"{row['mode']:5s} {row['status']} | {row.get('reason', '')}")
    print(f"risultati salvati in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
