"""Facade minimale per usare il progetto come pacchetto importabile."""

try:
    from model.config import ModelConfig, default_small
except Exception:  # pragma: no cover
    ModelConfig = None
    default_small = None

__all__ = ["ModelConfig", "default_small"]
