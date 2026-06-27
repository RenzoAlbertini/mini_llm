from .config import ModelConfig, default_small

__all__ = ["ModelConfig", "MiniTransformerLM", "default_small"]


def __getattr__(name):
    if name == "MiniTransformerLM":
        from .transformer import MiniTransformerLM

        return MiniTransformerLM
    raise AttributeError(name)
