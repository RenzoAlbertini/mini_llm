import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from inference.generate import generate, load_model, parse_stop_sequences
from tokenizer.tokenizer import BPETokenizer
from utils.helpers import get_device, set_seed


def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"benchmark dataset non trovato: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError("dataset supportati: .jsonl o .csv")


def safe_result_name(checkpoint_path):
    raw = str(checkpoint_path).replace("\\", "/")
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return raw.replace(".pt", "") or "checkpoint"


def choose_device(name):
    if name == "auto":
        return get_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesta ma non disponibile")
    return torch.device(name)


def continuation_token_ids(tokenizer, prompt, expected):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    full_ids = tokenizer.encode(prompt + expected, add_bos=True, add_eos=True)
    target_start = min(len(prompt_ids), len(full_ids) - 1)
    return full_ids, target_start


@torch.no_grad()
def score_expected(model, tokenizer, prompt, expected, device):
    if not expected:
        return None
    ids, target_start = continuation_token_ids(tokenizer, prompt, expected)
    if len(ids) < 2 or target_start >= len(ids):
        return None
    max_len = model.config.seq_len
    input_ids = ids[:-1]
    target_ids = ids[1:]
    target_mask = [idx >= target_start - 1 for idx in range(len(target_ids))]
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        input_ids = input_ids[overflow:]
        target_ids = target_ids[overflow:]
        target_mask = target_mask[overflow:]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    y = torch.tensor([target_ids], dtype=torch.long, device=device)
    mask = torch.tensor(target_mask, dtype=torch.bool, device=device)
    logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    selected = gathered[0][mask]
    if selected.numel() == 0:
        return None
    preds = logits.argmax(dim=-1)[0][mask]
    targets = y[0][mask]
    avg_log_likelihood = selected.mean().item()
    nll = -avg_log_likelihood
    return {
        "tokens": int(selected.numel()),
        "avg_log_likelihood": avg_log_likelihood,
        "perplexity": math.exp(min(nll, 20.0)),
        "token_accuracy": (preds == targets).float().mean().item(),
    }


def words(text):
    return re.findall(r"[A-Za-zÀ-ÿ0-9']+", text.lower())


def repetition_score(text):
    items = words(text)
    if len(items) < 2:
        return 1.0
    repeated_unigrams = 1.0 - (len(set(items)) / max(1, len(items)))
    bigrams = list(zip(items, items[1:]))
    repeated_bigrams = 0.0 if not bigrams else 1.0 - (len(set(bigrams)) / len(bigrams))
    return max(0.0, min(1.0, 1.0 - (0.6 * repeated_unigrams + 0.4 * repeated_bigrams)))


def coherence_score(prompt, response):
    response_words = words(response)
    if not response_words:
        return 0.0
    prompt_words = {w for w in words(prompt) if len(w) > 3}
    response_set = set(response_words)
    overlap = len(prompt_words & response_set) / max(1, min(len(prompt_words), 8))
    length_score = min(1.0, len(response_words) / 24.0)
    alpha_ratio = sum(ch.isalpha() or ch.isspace() or ch in ".,;:!?'-" for ch in response) / max(1, len(response))
    punctuation_bonus = 1.0 if any(ch in response for ch in ".!?") else 0.65
    repetition = repetition_score(response)
    raw = 0.30 * overlap + 0.25 * length_score + 0.20 * alpha_ratio + 0.15 * punctuation_bonus + 0.10 * repetition
    return round(max(0.0, min(1.0, raw)), 4)


def strip_prompt(generated, prompt):
    if generated.startswith(prompt):
        return generated[len(prompt):].strip()
    return generated.strip()


def aggregate(values, key):
    nums = [float(row[key]) for row in values if row.get(key) is not None]
    return sum(nums) / len(nums) if nums else None


def evaluate_checkpoint(
    checkpoint,
    dataset_path="benchmark/dataset.jsonl",
    tokenizer_path="tokenizer/tokenizer.json",
    out_dir="data/benchmarks",
    device_name="auto",
    max_samples=0,
    max_new_tokens=64,
    temperature=0.7,
    top_k=40,
    top_p=0.92,
    seed=42,
):
    set_seed(seed)
    started = time.perf_counter()
    checkpoint = Path(checkpoint)
    device = choose_device(device_name)
    tokenizer = BPETokenizer.load_model(tokenizer_path)
    model = load_model(str(checkpoint), device, quantized=False)
    model.eval()
    dataset = load_dataset(dataset_path)
    if max_samples and max_samples > 0:
        dataset = dataset[:max_samples]
    stop_sequences = parse_stop_sequences(tokenizer, ["\n\n", "\nQuestion:", "\nInstruction:"])
    samples = []
    scored = []
    for item in dataset:
        prompt = item.get("prompt", "")
        expected = item.get("expected") or item.get("target") or item.get("reference") or ""
        item_max_tokens = int(item.get("max_new_tokens") or max_new_tokens)
        generated = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=item_max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_sequences=stop_sequences,
            repetition_penalty=1.08,
            device=device,
        )
        response = strip_prompt(generated, prompt)
        expected_score = score_expected(model, tokenizer, prompt, expected, device)
        row = {
            "id": item.get("id"),
            "category": item.get("category"),
            "prompt": prompt,
            "expected": expected,
            "response": response,
            "coherence_score": coherence_score(prompt, response),
            "repetition_score": repetition_score(response),
        }
        if expected_score:
            row.update(expected_score)
            scored.append(row)
        samples.append(row)
    metrics = {
        "perplexity": aggregate(scored, "perplexity"),
        "average_log_likelihood": aggregate(scored, "avg_log_likelihood"),
        "token_accuracy": aggregate(scored, "token_accuracy"),
        "response_coherence_score": aggregate(samples, "coherence_score"),
        "repetition_penalty_score": aggregate(samples, "repetition_score"),
        "scored_items": len(scored),
        "generated_items": len(samples),
    }
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "dataset": str(dataset_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.perf_counter() - started, 3),
        "device": str(device),
        "metrics": metrics,
        "samples": samples,
    }
    out_path = Path(out_dir) / f"results_{safe_result_name(checkpoint)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["result_path"] = str(out_path)
    return payload


def main():
    parser = argparse.ArgumentParser(description="MiniLLM Benchmark Suite.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best.pt")
    parser.add_argument("--dataset", default="benchmark/dataset.jsonl")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--out_dir", default="data/benchmarks")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        device_name=args.device,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(json.dumps({"result_path": result["result_path"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
