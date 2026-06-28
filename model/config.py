from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    seq_len: int = 256
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    bias: bool = True
    use_gradient_checkpointing: bool = False

    def to_dict(self):
        values = asdict(self)
        values["hidden_size"] = self.d_model
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "hidden_size" in values and "d_model" not in values:
            values["d_model"] = values.pop("hidden_size")
        else:
            values.pop("hidden_size", None)
        return cls(**values)

    @property
    def hidden_size(self):
        return self.d_model


def default_small():
    """Config piccola pensata per prove rapide e GPU tipo RTX 3050."""
    return ModelConfig(
        vocab_size=8192,
        seq_len=128,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        dropout=0.1,
        use_gradient_checkpointing=False,
    )


def mini_llm_32m(vocab_size=8192):
    """Preset educativo piu grande, pensato per RTX 3050 8 GB."""
    return ModelConfig(
        vocab_size=vocab_size,
        seq_len=512,
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_ff=1536,
        dropout=0.1,
        use_gradient_checkpointing=True,
    )
