import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_step(name, command, required=False):
    print(f"\n== {name} ==")
    started = time.perf_counter()
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except Exception as exc:
        return {
            "name": name,
            "status": "FAIL",
            "seconds": 0.0,
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
            "required": required,
        }

    elapsed = time.perf_counter() - started
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

    status = "OK" if result.returncode == 0 else ("FAIL" if required else "SKIP")
    return {
        "name": name,
        "status": status,
        "seconds": elapsed,
        "returncode": result.returncode,
        "command": command,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
        "required": required,
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline end-to-end mini_llm.")
    parser.add_argument("--out", default="data/pipeline/status.json")
    parser.add_argument("--checkpoint", default="models/checkpoints/demo/final.pt")
    args = parser.parse_args()

    py = sys.executable
    steps = [
        ("training", [py, "-m", "run_training", "--demo", "--max_steps", "8"], True),
        ("evaluation", [py, "-m", "evaluate_model", "--checkpoint", args.checkpoint, "--tokenizer", "data/processed/demo_tokenizer.json", "--data_dir", "data/processed/demo_dataset.txt", "--processed", "data/processed/pipeline_eval_tokens.pt"], True),
        ("export", [py, "-m", "export_model", "--checkpoint", args.checkpoint, "--out_dir", "export"], True),
        ("benchmark", [py, "-m", "benchmark_inference", "--checkpoint", args.checkpoint, "--tokenizer", "data/processed/demo_tokenizer.json", "--max_new_tokens", "16"], True),
        ("generate_demo", [py, "-m", "run_generate", "--checkpoint", args.checkpoint, "--tokenizer", "data/processed/demo_tokenizer.json", "--prompt", "python is", "--max_new_tokens", "24", "--mode", "production"], True),
    ]

    started = time.perf_counter()
    results = [run_step(name, command, required) for name, command, required in steps]
    ok = all(row["status"] == "OK" or not row["required"] for row in results)
    payload = {
        "status": "OK" if ok else "FAIL",
        "seconds": time.perf_counter() - started,
        "steps": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nPIPELINE REPORT")
    print("===============")
    for row in results:
        print(f"{row['name']:18s} {row['status']:5s} {row['seconds']:.2f}s")
    print(f"summary: {payload['status']}")
    print(f"stato salvato in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
