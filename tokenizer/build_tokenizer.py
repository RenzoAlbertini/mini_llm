import argparse
from collections import Counter
from pathlib import Path

from tokenizer.tokenizer import BPETokenizer


def iter_text_files(data_dir):
    data_dir = Path(data_dir)
    for path in sorted(data_dir.glob("**/*.txt")):
        if path.is_file():
            yield path


def read_corpus(data_dir):
    texts = []
    for path in iter_text_files(data_dir):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    if not texts:
        raise FileNotFoundError(f"Nessun file .txt trovato in {data_dir}")
    return "\n".join(texts)


def train_byte_bpe(text, vocab_size=8192, min_frequency=2):
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

        if (step + 1) % 100 == 0:
            print(f"merge {step + 1}/{max_merges} - freq={freq} - vocab={next_id}")

    tokenizer.merges = merges
    tokenizer.merge_map = {(a, b): c for a, b, c in merges}
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Addestra un tokenizer BPE byte-level minimale.")
    parser.add_argument("--data_dir", default="data/raw", help="Cartella con file .txt grezzi.")
    parser.add_argument("--out", default="tokenizer/tokenizer.json", help="Path del tokenizer salvato.")
    parser.add_argument("--vocab_size", type=int, default=8192, help="Dimensione massima del vocabolario.")
    parser.add_argument("--min_frequency", type=int, default=2, help="Frequenza minima per un merge.")
    args = parser.parse_args()

    text = read_corpus(args.data_dir)
    tokenizer = train_byte_bpe(text, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    tokenizer.save_model(args.out)
    print(f"Tokenizer salvato in {args.out} con vocab_size={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
