import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_checkpoint(path, model, optimizer=None, config=None, step=0, val_loss=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    torch.save(payload, path)


def load_checkpoint(path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def save_weights(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(path, model, device="cpu"):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


@contextmanager
def timer(name="tempo"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.2f}s")


def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def memory_usage(device=None):
    device = device or get_device()
    if device.type != "cuda":
        return {"device": str(device), "allocated": "n/a", "reserved": "n/a"}
    return {
        "device": str(device),
        "allocated": format_bytes(torch.cuda.memory_allocated(device)),
        "reserved": format_bytes(torch.cuda.memory_reserved(device)),
        "max_allocated": format_bytes(torch.cuda.max_memory_allocated(device)),
    }


def get_amp_dtype():
    if not torch.cuda.is_available():
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def benchmark_gpu(device=None, matrix_size=1024, steps=20):
    device = device or get_device()
    if device.type != "cuda":
        return {"device": str(device), "message": "CUDA non disponibile"}

    dtype = get_amp_dtype() or torch.float16
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "device": torch.cuda.get_device_name(device),
        "dtype": str(dtype).replace("torch.", ""),
        "matrix_size": matrix_size,
        "steps": steps,
        "seconds": round(elapsed, 4),
        "steps_per_second": round(steps / elapsed, 2),
        "memory": memory_usage(device),
    }


def gpu_profile(model, sample_batch, out_dir="data/profiling", steps=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    if device.type != "cuda":
        report = out_dir / "profile.txt"
        report.write_text("CUDA non disponibile: profiling GPU saltato.\n", encoding="utf-8")
        return report

    inputs, targets = sample_batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    model.train()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(steps):
            _, loss = model(inputs, targets)
            loss.backward()
            model.zero_grad(set_to_none=True)
            prof.step()

    report = out_dir / "profile.txt"
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    report.write_text(table, encoding="utf-8")
    return report
