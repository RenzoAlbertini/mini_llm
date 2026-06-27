import json
from pathlib import Path


class BPETokenizer:
    """Byte-level BPE tokenizer minimale.

    Il vocabolario parte dai 256 byte possibili e aggiunge merge appresi.
    Questo lo rende robusto su qualunque testo UTF-8 senza dipendenze esterne.
    """

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, merges=None, vocab=None):
        self.special_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}
        self.byte_offset = len(self.SPECIAL_TOKENS)

        if vocab is None:
            self.vocab = {
                i + self.byte_offset: bytes([i])
                for i in range(256)
            }
        else:
            self.vocab = {int(k): bytes.fromhex(v) for k, v in vocab.items()}

        self.merges = [(int(a), int(b), int(c)) for a, b, c in (merges or [])]
        self.merge_map = {(a, b): c for a, b, c in self.merges}

    @property
    def pad_id(self):
        return self.special_to_id["<pad>"]

    @property
    def bos_id(self):
        return self.special_to_id["<bos>"]

    @property
    def eos_id(self):
        return self.special_to_id["<eos>"]

    @property
    def unk_id(self):
        return self.special_to_id["<unk>"]

    @property
    def vocab_size(self):
        return max(self.vocab.keys(), default=self.byte_offset - 1) + 1

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [byte + self.byte_offset for byte in text.encode("utf-8")]

        for left, right, merged in self.merges:
            ids = self._replace_pair(ids, left, right, merged)

        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special=True):
        chunks = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self.id_to_special:
                if not skip_special:
                    chunks.append(self.id_to_special[token_id].encode("utf-8"))
                continue
            chunks.append(self.vocab.get(token_id, b""))
        return b"".join(chunks).decode("utf-8", errors="replace")

    def test_roundtrip(self, text):
        """Ritorna True se decode(encode(text)) ricostruisce il testo."""
        return self.decode(self.encode(text)) == text

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "byte_bpe",
            "special_tokens": self.SPECIAL_TOKENS,
            "merges": self.merges,
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_model(self, path):
        """Alias esplicito per salvare il tokenizer su disco."""
        self.save(path)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=payload["merges"], vocab=payload["vocab"])

    @classmethod
    def load_model(cls, path):
        """Alias esplicito per caricare il tokenizer da disco."""
        return cls.load(path)

    @staticmethod
    def _replace_pair(ids, left, right, merged):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == left and ids[i + 1] == right:
                out.append(merged)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out


def test_roundtrip(text="ciao mondo", tokenizer=None):
    tokenizer = tokenizer or BPETokenizer()
    return tokenizer.test_roundtrip(text)
