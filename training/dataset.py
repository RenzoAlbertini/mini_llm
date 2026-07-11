from array import array
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, random_split

from tokenizer.tokenizer import BPETokenizer


def parse_weighted_sources(sources):
    parsed = []
    for source in sources or []:
        if ":" in source:
            path, weight = source.rsplit(":", 1)
            try:
                parsed.append((Path(path), float(weight)))
                continue
            except ValueError:
                pass
        parsed.append((Path(source), 1.0))
    return parsed


def iter_text_files(data_dir):
    data_dir = Path(data_dir)
    if data_dir.is_file():
        yield data_dir
        return
    for path in sorted(data_dir.glob("**/*.txt")):
        if path.is_file():
            yield path


def load_raw_text(data_dir, sources=None):
    if sources:
        chunks = []
        for path, weight in parse_weighted_sources(sources):
            repeat = max(1, int(round(weight)))
            for text_path in iter_text_files(path):
                text = text_path.read_text(encoding="utf-8", errors="ignore")
                chunks.extend([text] * repeat)
        if not chunks:
            raise FileNotFoundError("Nessun testo trovato nelle sorgenti indicate")
        return "\n".join(chunks)

    paths = list(iter_text_files(data_dir))
    if not paths:
        raise FileNotFoundError(f"Nessun file .txt trovato in {data_dir}")
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths)


def load_or_tokenize(data_dir, tokenizer_path, processed_path, add_eos=True, sources=None, max_chars=0):
    processed_path = Path(processed_path)
    source_paths = []
    if sources:
        for path, _ in parse_weighted_sources(sources):
            source_paths.extend(iter_text_files(path))
    else:
        source_paths.extend(iter_text_files(data_dir))
    fingerprint_data = {
        "version": 2,
        "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        "sources": [
            [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]
            for path in source_paths
        ],
        "weighted_sources": list(sources or []),
        "add_eos": bool(add_eos),
        "max_chars": int(max_chars),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if processed_path.exists():
        cached = torch.load(processed_path, map_location="cpu")
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            return cached["tokens"]
        print("cache token non valida: dataset o tokenizer modificato, rigenero", flush=True)

    tokenizer = BPETokenizer.load_model(tokenizer_path)
    text = load_raw_text(data_dir, sources=sources)
    if max_chars and len(text) > max_chars:
        print(f"tokenizzazione: uso {max_chars:,} caratteri su {len(text):,}", flush=True)
        text = text[:max_chars]
    ids = tokenizer.encode(text, add_eos=add_eos)
    tokens = torch.tensor(ids, dtype=torch.long)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": tokens, "fingerprint": fingerprint}, processed_path)
    return tokens


def tokenize_to_binary(data_dir, tokenizer_path, out_path, add_eos=True):
    """Tokenizza e salva token uint32 in un file binario semplice.

    Utile quando il corpus cresce e non vuoi tenere anche la lista Python
    intermedia salvata in un checkpoint `.pt`.
    """
    tokenizer = BPETokenizer.load_model(tokenizer_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for path in sorted(Path(data_dir).glob("**/*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            ids = tokenizer.encode(text, add_eos=add_eos)
            array("I", ids).tofile(f)
    return out_path


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len, stride=1):
        if len(tokens) <= seq_len:
            raise ValueError("Il corpus tokenizzato deve essere piu lungo di seq_len.")
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = max(1, stride)

    def __len__(self):
        return max(1, (len(self.tokens) - self.seq_len) // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.tokens[start:start + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class BinaryTokenDataset(Dataset):
    """Dataset lazy su file binario uint32 creato da tokenize_to_binary()."""

    def __init__(self, token_file, seq_len, stride=1):
        self.token_file = Path(token_file)
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.length = self.token_file.stat().st_size // array("I").itemsize
        if self.length <= seq_len:
            raise ValueError("Il file tokenizzato deve essere piu lungo di seq_len.")

    def __len__(self):
        return max(1, (self.length - self.seq_len) // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        values = array("I")
        with self.token_file.open("rb") as f:
            f.seek(start * values.itemsize)
            values.fromfile(f, self.seq_len + 1)
        chunk = torch.tensor(values, dtype=torch.long)
        return chunk[:-1], chunk[1:]


class WeightedMultiDataset(Dataset):
    """Combina dataset tokenizzati con priorita diverse.

    Per restare semplice usa un ConcatDataset e fornisce anche pesi per
    WeightedRandomSampler quando vuoi campionamento bilanciato.
    """

    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        self.concat = ConcatDataset(datasets)

    def __len__(self):
        return len(self.concat)

    def __getitem__(self, idx):
        return self.concat[idx]

    def sample_weights(self):
        values = []
        for dataset, weight in zip(self.datasets, self.weights):
            values.extend([float(weight)] * len(dataset))
        return torch.tensor(values, dtype=torch.double)


def create_weighted_dataloader(datasets, weights, batch_size, shuffle=True, num_workers=0):
    dataset = WeightedMultiDataset(datasets, weights)
    sampler = None
    if shuffle:
        sampler = WeightedRandomSampler(dataset.sample_weights(), num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def split_dataset(dataset, val_fraction=0.05, seed=42):
    if len(dataset) < 2:
        return dataset, dataset
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def create_dataloaders(tokens, seq_len, batch_size, val_fraction=0.05, stride=1, seed=42, num_workers=0):
    """Crea split contigui prima delle finestre per evitare data leakage.

    Con stride piccolo, finestre adiacenti condividono quasi tutti i token. Uno
    split casuale delle finestre renderebbe quindi la validation artificialmente
    facile. La fascia di ``seq_len`` token tra gli split impedisce sovrapposizioni.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction deve essere compresa tra 0 e 1")
    min_tokens = 2 * (seq_len + 1) + seq_len
    if len(tokens) < min_tokens:
        raise ValueError(
            f"Servono almeno {min_tokens} token per split train/validation indipendenti; "
            f"trovati {len(tokens)}. Riduci seq_len o usa piu dati."
        )

    split_at = int(len(tokens) * (1.0 - val_fraction))
    split_at = min(split_at, len(tokens) - seq_len - 1)
    train_tokens = tokens[:split_at]
    val_tokens = tokens[split_at + seq_len:]
    train_ds = TokenDataset(train_tokens, seq_len=seq_len, stride=stride)
    val_ds = TokenDataset(val_tokens, seq_len=seq_len, stride=stride)

    drop_last = len(train_ds) >= batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
