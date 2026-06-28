import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import ModelConfig
from model.transformer import MiniTransformerLM
from utils.helpers import get_device, load_checkpoint


def pack_int4(values):
    values = values.to(torch.uint8).flatten()
    if values.numel() % 2 == 1:
        values = torch.cat([values, torch.zeros(1, dtype=torch.uint8)])
    high = values[0::2] << 4
    low = values[1::2] & 0x0F
    return high | low


def unpack_int4(packed, numel):
    packed = packed.to(torch.uint8).flatten()
    high = packed >> 4
    low = packed & 0x0F
    values = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    values[0::2] = high
    values[1::2] = low
    return values[:numel]


def quantize_tensor_int4(tensor):
    tensor = tensor.detach().cpu().float()
    max_abs = tensor.abs().max()
    scale = max_abs / 7.0 if max_abs > 0 else torch.tensor(1.0)
    q = torch.clamp(torch.round(tensor / scale), -8, 7).to(torch.int8)
    shifted = (q + 8).to(torch.uint8)
    return {
        "packed": pack_int4(shifted),
        "shape": tuple(tensor.shape),
        "numel": tensor.numel(),
        "scale": float(scale),
    }


def dequantize_tensor_int4(entry, dtype=torch.float32):
    shifted = unpack_int4(entry["packed"], entry["numel"]).to(torch.int16)
    q = (shifted - 8).to(torch.float32)
    return (q * float(entry["scale"])).view(entry["shape"]).to(dtype)


def bitsandbytes_available():
    try:
        import bitsandbytes as _bnb  # noqa: F401

        return True
    except Exception:
        return False


def save_4bit_model(model, out_path, config=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backend = "bitsandbytes-nf4-compatible" if bitsandbytes_available() else "portable-int4"
    q_state = {}
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            q_state[name] = quantize_tensor_int4(tensor)
        else:
            q_state[name] = {"raw": tensor.detach().cpu()}
    payload = {
        "quantization": "4bit",
        "backend": backend,
        "model": q_state,
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config or {}),
    }
    torch.save(payload, out_path)
    return out_path


def load_4bit_state_dict(path, dtype=torch.float32, device="cpu"):
    payload = torch.load(path, map_location="cpu")
    state_dict = {}
    for name, entry in payload["model"].items():
        if "raw" in entry:
            state_dict[name] = entry["raw"].to(device)
        else:
            state_dict[name] = dequantize_tensor_int4(entry, dtype=dtype).to(device)
    return state_dict, payload.get("config", {})


def load_4bit_model(path, model, dtype=torch.float32, device="cpu"):
    state_dict, config = load_4bit_state_dict(path, dtype=dtype, device=device)
    model.load_state_dict(state_dict)
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Quantizza MiniLLM in 4-bit e salva un checkpoint locale.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best.pt")
    parser.add_argument("--out", default="models/quantized/mini_llm_32m_4bit.pt")
    args = parser.parse_args()

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    config = ModelConfig.from_dict(checkpoint["config"])
    model = MiniTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    out_path = save_4bit_model(model, args.out, config=config)
    backend = "bitsandbytes presente" if bitsandbytes_available() else "fallback int4 portabile"
    print(f"checkpoint 4-bit salvato: {out_path} ({backend})")


if __name__ == "__main__":
    main()
