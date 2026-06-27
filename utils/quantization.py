from pathlib import Path

import torch


def quantize_tensor(tensor):
    tensor = tensor.detach().cpu().float()
    max_abs = tensor.abs().max()
    scale = max_abs / 127.0 if max_abs > 0 else torch.tensor(1.0)
    q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return q, float(scale)


def dequantize_tensor(q_tensor, scale, dtype=torch.float32):
    return (q_tensor.float() * scale).to(dtype)


def save_quantized_model(model, path, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q_state = {}
    scales = {}
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            q_tensor, scale = quantize_tensor(tensor)
            q_state[name] = q_tensor
            scales[name] = scale
        else:
            q_state[name] = tensor.cpu()
    payload = {"model": q_state, "scales": scales}
    if config is not None:
        payload["config"] = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    torch.save(payload, path)


def load_quantized_state_dict(path, dtype=torch.float32, device="cpu"):
    payload = torch.load(path, map_location="cpu")
    state_dict = {}
    scales = payload.get("scales", {})
    for name, tensor in payload["model"].items():
        if name in scales:
            state_dict[name] = dequantize_tensor(tensor, scales[name], dtype=dtype).to(device)
        else:
            state_dict[name] = tensor.to(device)
    return state_dict, payload.get("config")


def load_quantized_model(path, model, dtype=torch.float32, device="cpu"):
    state_dict, config = load_quantized_state_dict(path, dtype=dtype, device=device)
    model.load_state_dict(state_dict)
    return model, config
