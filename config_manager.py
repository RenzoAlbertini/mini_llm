import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from model.config import ModelConfig


MODEL_PRESETS = {
    "small": ModelConfig(seq_len=128, d_model=256, n_layers=4, n_heads=4, d_ff=1024, dropout=0.1),
    "medium": ModelConfig(seq_len=256, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1),
    "large": ModelConfig(seq_len=512, d_model=768, n_layers=10, n_heads=12, d_ff=3072, dropout=0.1),
}


@dataclass
class TrainingConfig:
    preset: str = "small"
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    scheduler: str = "cosine"
    weight_decay: float = 0.1
    batch_size: int = 4
    epochs: int = 1
    max_steps: int = 200
    eval_every: int = 50
    save_every: int = 0
    patience: int = 0
    curriculum: bool = False
    curriculum_start_seq_len: int = 64
    curriculum_step_size: int = 200
    curriculum_increment: int = 64

    def to_dict(self):
        return asdict(self)


def add_config_args(parser):
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="small")
    parser.add_argument("--config", default=None, help="JSON opzionale con model/training config.")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum_start_seq_len", type=int, default=None)
    parser.add_argument("--curriculum_step_size", type=int, default=None)
    parser.add_argument("--curriculum_increment", type=int, default=None)


def load_config_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_values = payload.get("model", payload)
    training_values = payload.get("training", {})
    return model_values, training_values


def build_configs(args):
    model_config = MODEL_PRESETS[getattr(args, "preset", "small")].__class__(
        **MODEL_PRESETS[getattr(args, "preset", "small")].to_dict()
    )
    training_config = TrainingConfig(preset=getattr(args, "preset", "small"))

    if getattr(args, "config", None):
        model_values, training_values = load_config_file(args.config)
        model_config = ModelConfig.from_dict({**model_config.to_dict(), **model_values})
        training_config = TrainingConfig(**{**training_config.to_dict(), **training_values})

    for name in ["seq_len", "d_model", "n_layers", "n_heads", "d_ff", "dropout", "vocab_size"]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(model_config, name, value)

    for name in [
        "scheduler",
        "warmup_steps",
        "min_lr",
        "curriculum_start_seq_len",
        "curriculum_step_size",
        "curriculum_increment",
    ]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(training_config, name, value)
    if getattr(args, "curriculum", False):
        training_config.curriculum = True

    validate_model_config(model_config)
    return model_config, training_config


def validate_model_config(config):
    if config.vocab_size <= 0:
        raise ValueError("vocab_size deve essere positivo")
    if config.seq_len <= 0:
        raise ValueError("seq_len deve essere positivo")
    if config.d_model <= 0 or config.d_ff <= 0:
        raise ValueError("d_model e d_ff devono essere positivi")
    if config.n_layers <= 0 or config.n_heads <= 0:
        raise ValueError("n_layers e n_heads devono essere positivi")
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model deve essere divisibile per n_heads")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout deve essere in [0, 1)")


def save_run_config(path, model_config, training_config=None):
    payload = {"model": model_config.to_dict()}
    if training_config is not None:
        payload["training"] = training_config.to_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_parser(description="mini_llm config"):
    parser = argparse.ArgumentParser(description=description)
    add_config_args(parser)
    return parser
