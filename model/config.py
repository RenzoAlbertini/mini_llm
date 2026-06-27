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
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


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
