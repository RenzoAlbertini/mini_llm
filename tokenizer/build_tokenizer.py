import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import BPETokenizer


def iter_text_files(data_dir):
    data_dir = Path(data_dir)
    for path in sorted(data_dir.glob("**/*.txt")):
        if path.is_file():
            yield path


def read_corpus(data_dir):
    data_dir = Path(data_dir)
    large_dataset = data_dir / "dataset_large.txt"
    if large_dataset.exists():
        print(f"Uso dataset grande: {large_dataset}")
        return large_dataset.read_text(encoding="utf-8", errors="ignore")
    texts = []
    for path in iter_text_files(data_dir):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    if not texts:
        raise FileNotFoundError(f"Nessun file .txt trovato in {data_dir}")
    return "\n".join(texts)


def train_byte_bpe(text, vocab_size=8192, min_frequency=2, progress_every=25):
    tokenizer = BPETokenizer()
    ids = [byte + tokenizer.byte_offset for byte in text.encode("utf-8")]
    merges = []
    next_id = tokenizer.byte_offset + 256

    max_merges = max(0, vocab_size - next_id)
    for step in range(max_merges):
        pair_counts = Counter(zip(ids, ids[1:]))
        if not pair_counts:
            break

        (left, right), freq = pair_counts.most_common(1)[0]
        if freq < min_frequency:
            break

        merged = next_id
        next_id += 1
        merges.append((left, right, merged))
        tokenizer.vocab[merged] = tokenizer.vocab[left] + tokenizer.vocab[right]
        ids = tokenizer._replace_pair(ids, left, right, merged)

        if (step + 1) % max(1, progress_every) == 0:
            print(f"merge {step + 1}/{max_merges} - freq={freq} - vocab={next_id}", flush=True)

    tokenizer.merges = merges
    tokenizer.merge_map = {(a, b): c for a, b, c in merges}
    return tokenizer


def export_vocab_and_merges(tokenizer, out_path):
    out_path = Path(out_path)
    vocab_path = out_path.with_name("vocab.json")
    merges_path = out_path.with_name("merges.txt")
    vocab = {}
    for token_id, token_bytes in sorted(tokenizer.vocab.items()):
        try:
            token_text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_text = token_bytes.hex()
        vocab[str(token_id)] = token_text
    vocab_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    merges_lines = [f"{left} {right} {merged}" for left, right, merged in tokenizer.merges]
    merges_path.write_text("\n".join(merges_lines) + "\n", encoding="utf-8")
    return vocab_path, merges_path


def main():
    parser = argparse.ArgumentParser(description="Addestra un tokenizer BPE byte-level minimale.")
    parser.add_argument("--data_dir", default="data/raw", help="Cartella con file .txt grezzi.")
    parser.add_argument("--input_file", default=None, help="File testo singolo, ad esempio data/raw/dataset_large.txt.")
    parser.add_argument("--dataset", default=None, help="Alias per --input_file.")
    parser.add_argument("--out", default="tokenizer/tokenizer.json", help="Path del tokenizer salvato.")
    parser.add_argument("--vocab_size", type=int, default=8192, help="Dimensione massima del vocabolario.")
    parser.add_argument("--min_frequency", type=int, default=2, help="Frequenza minima per un merge.")
    parser.add_argument(
        "--max_chars",
        type=int,
        default=250_000,
        help="Numero massimo di caratteri usati per addestrare il BPE. Usa 0 per tutto il dataset.",
    )
    parser.add_argument("--progress_every", type=int, default=25, help="Stampa progresso ogni N merge.")
    args = parser.parse_args()

    dataset_path = args.dataset or args.input_file
    text = Path(dataset_path).read_text(encoding="utf-8", errors="ignore") if dataset_path else read_corpus(args.data_dir)
    original_chars = len(text)
    if args.max_chars and len(text) > args.max_chars:
        print(
            f"Dataset grande: uso i primi {args.max_chars:,} caratteri su {original_chars:,}. "
            "Passa --max_chars 0 per usare tutto.",
            flush=True,
        )
        text = text[:args.max_chars]
    print(f"Training tokenizer: chars={len(text):,} | vocab_size={args.vocab_size}", flush=True)
    tokenizer = train_byte_bpe(
        text,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        progress_every=args.progress_every,
    )
    tokenizer.save_model(args.out)
    vocab_path, merges_path = export_vocab_and_merges(tokenizer, args.out)
    print(f"Tokenizer salvato in {args.out} con vocab_size={tokenizer.vocab_size}")
    print(f"Vocab salvato in {vocab_path}")
    print(f"Merges salvato in {merges_path}")


if __name__ == "__main__":
    main()
