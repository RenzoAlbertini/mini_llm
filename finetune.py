import argparse
import csv
import gzip
import json
import shutil
import urllib.request
from pathlib import Path

from training.train import train_model


DAILYDIALOG_URL = "https://raw.githubusercontent.com/declare-lab/conv-emotion/master/dailydialog/dailydialog.csv"
OPENASSISTANT_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_all.messages.jsonl.gz"


def fetch_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": "mini_llm_finetune/2.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        if url.endswith(".gz"):
            payload = gzip.decompress(payload)
        return payload.decode("utf-8", errors="ignore")


def normalize_turn(text):
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def dialogue_pairs_from_turns(turns):
    lines = []
    cleaned = [normalize_turn(turn) for turn in turns if normalize_turn(turn)]
    for i in range(0, len(cleaned) - 1, 2):
        lines.append(f"User: {cleaned[i]}\nAssistant: {cleaned[i + 1]}")
    return lines


def parse_dailydialog(raw):
    pairs = []
    for row in csv.reader(raw.splitlines()):
        joined = " ".join(row)
        if "__eou__" in joined:
            pairs.extend(dialogue_pairs_from_turns(joined.split("__eou__")))
    return pairs


def parse_openassistant(raw):
    pairs = []
    last_user = None
    for line in raw.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = normalize_turn(item.get("text") or item.get("message", {}).get("content"))
        role = str(item.get("role") or item.get("author", {}).get("role") or "").lower()
        if not text:
            continue
        if "assistant" in role and last_user:
            pairs.append(f"User: {last_user}\nAssistant: {text}")
            last_user = None
        elif "prompter" in role or "user" in role:
            last_user = text
    return pairs


def prepare_dialogue_dataset(dataset_dialogue, out_dir="data/raw/dialogue", max_pairs=2000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dialogue_dataset.txt"
    pairs = []

    source_path = Path(dataset_dialogue)
    if source_path.exists():
        if source_path.is_dir():
            for text_path in sorted(source_path.glob("**/*")):
                if text_path.suffix.lower() in {".txt", ".jsonl", ".csv"}:
                    pairs.append(text_path.read_text(encoding="utf-8", errors="ignore"))
        else:
            pairs.append(source_path.read_text(encoding="utf-8", errors="ignore"))
    else:
        wanted = {dataset_dialogue.lower()}
        if "both" in wanted:
            wanted = {"dailydialog", "openassistant"}
        if "dailydialog" in wanted:
            pairs.extend(parse_dailydialog(fetch_text(DAILYDIALOG_URL)))
        if "openassistant" in wanted:
            pairs.extend(parse_openassistant(fetch_text(OPENASSISTANT_URL)))

    if not pairs:
        raise RuntimeError(
            "Nessun dialogo trovato. Usa --dataset_dialogue dailydialog, openassistant, both "
            "oppure passa un path locale a file/cartella."
        )
    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    out_path.write_text("\n\n".join(pairs) + "\n", encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning conversazionale locale di mini_llm.")
    parser.add_argument("--base_checkpoint", default="models/checkpoints/mini_llm_32m_best.pt")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--processed", default="data/processed/finetune_tokens.pt")
    parser.add_argument("--checkpoint_dir", default="models/checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=20)
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
    parser.add_argument(
        "--dataset_dialogue",
        default=None,
        help="dailydialog, openassistant, both, oppure path locale a file/cartella di dialoghi.",
    )
    parser.add_argument("--max_dialogue_pairs", type=int, default=2000)
    parser.add_argument("--quantized_base", action="store_true", help="Carica base 8-bit dequantizzata prima del fine-tuning.")
    parser.add_argument("--save_quantized", action="store_true")
    parser.add_argument("--stats_path", default="data/logs/finetune_stats.csv")
    parser.add_argument("--out", default="models/checkpoints/mini_llm_32m_finetuned.pt")
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

    if args.dataset_dialogue:
        dialogue_path = prepare_dialogue_dataset(args.dataset_dialogue, max_pairs=args.max_dialogue_pairs)
        args.data_dir = str(dialogue_path)
        args.processed = "data/processed/dialogue_tokens.pt"
        print(f"dataset conversazionale: {dialogue_path}")

    device = get_device()
    if not Path(args.base_checkpoint).exists():
        print(f"FAIL: checkpoint base non trovato: {args.base_checkpoint}")
        return 1
    base_model = load_model(args.base_checkpoint, device, quantized=args.quantized_base)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.checkpoint_dir) / "resume_from_base.pt"
    torch.save({"model": base_model.state_dict(), "config": base_model.config.to_dict(), "step": 0, "epoch": 0}, resume_path)
    args.resume = str(resume_path)

    tokenizer = BPETokenizer.load_model(args.tokenizer)
    config = base_model.config
    config.vocab_size = tokenizer.vocab_size
    final_path = train_model(args, config=config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_path, out_path)

    if args.save_quantized:
        model = load_model(final_path, device, quantized=False)
        save_quantized_model(model, Path(args.checkpoint_dir) / "final_quantized.pt", config=model.config)
    print(f"fine-tuning completato: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
