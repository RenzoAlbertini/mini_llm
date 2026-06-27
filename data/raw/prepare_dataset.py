import argparse
import re
import urllib.request
from pathlib import Path


SAMPLE_PARAGRAPHS = [
    "python is a programming language used for scripting, automation, data analysis and machine learning.",
    "pytorch is a tensor library for building neural networks with automatic differentiation.",
    "a transformer language model predicts the next token from the previous context.",
    "self attention lets each token compare itself with earlier tokens in the sequence.",
    "the training loop computes logits, cross entropy loss, gradients and optimizer updates.",
    "small models are useful for experiments because every tensor shape can be inspected.",
    "a dataset should contain enough varied text to teach the tokenizer common fragments.",
    "during inference the model samples one token at a time using temperature and filtering.",
]

SAMPLE_TEXT = "\n".join(SAMPLE_PARAGRAPHS * 12)


def clean_text(text, lowercase=True):
    if lowercase:
        text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9àèéìòùç.,;:!?'\-\n ]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def download_text(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def read_inputs(paths):
    texts = []
    for path in paths:
        texts.append(Path(path).read_text(encoding="utf-8", errors="ignore"))
    return texts


def read_sources(sources):
    texts = []
    for source in sources or []:
        if ":" in source:
            path, weight = source.rsplit(":", 1)
            try:
                repeat = max(1, int(round(float(weight))))
            except ValueError:
                path, repeat = source, 1
        else:
            path, repeat = source, 1
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.glob("**/*.txt"))
        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            texts.extend([text] * repeat)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Prepara un dataset testuale semplice per mini_llm.")
    parser.add_argument("--url", action="append", default=[], help="URL testuale da scaricare. Ripetibile.")
    parser.add_argument("--input", action="append", default=[], help="File locale da importare. Ripetibile.")
    parser.add_argument("--source", action="append", default=[], help="File/cartella con peso opzionale path:weight.")
    parser.add_argument("--out", default="data/raw/dataset.txt")
    parser.add_argument("--keep_case", action="store_true")
    args = parser.parse_args()

    texts = []
    for url in args.url:
        print(f"scarico: {url}")
        texts.append(download_text(url))
    texts.extend(read_inputs(args.input))
    texts.extend(read_sources(args.source))
    if not texts:
        texts.append(SAMPLE_TEXT)

    cleaned = "\n\n".join(clean_text(text, lowercase=not args.keep_case) for text in texts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cleaned, encoding="utf-8")
    print(f"dataset salvato in {out} ({len(cleaned)} caratteri)")


if __name__ == "__main__":
    main()
