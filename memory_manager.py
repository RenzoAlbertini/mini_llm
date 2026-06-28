from dataclasses import dataclass

import torch


@dataclass
class PrecisionDecision:
    use_amp: bool
    dtype: torch.dtype | None
    name: str


class MemoryManager:
    """Piccolo helper per restare dentro GPU consumer come RTX 3050 8 GB."""

    def __init__(self, device=None, reserve_fraction=0.12):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.reserve_fraction = reserve_fraction

    def cuda_available(self):
        return self.device.type == "cuda" and torch.cuda.is_available()

    def vram_stats(self):
        if not self.cuda_available():
            return {
                "available": False,
                "total": None,
                "free": None,
                "allocated": None,
                "reserved": None,
                "used_percent": None,
                "device_name": str(self.device),
            }
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        used_percent = int(round((1.0 - free / max(1, total)) * 100))
        return {
            "available": True,
            "total": int(total),
            "free": int(free),
            "allocated": int(allocated),
            "reserved": int(reserved),
            "used_percent": used_percent,
            "device_name": torch.cuda.get_device_name(self.device),
        }

    def choose_precision(self):
        if not self.cuda_available():
            return PrecisionDecision(False, None, "fp32")
        return PrecisionDecision(True, torch.float16, "fp16")

    def fallback_precision(self, current_dtype):
        if not self.cuda_available():
            return PrecisionDecision(False, None, "fp32")
        if current_dtype == torch.float16 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return PrecisionDecision(True, torch.bfloat16, "bf16")
        return PrecisionDecision(False, None, "fp32")

    def recommend_batch_size(self, requested_batch_size, seq_len, hidden_size, n_layers):
        if not self.cuda_available():
            return requested_batch_size
        stats = self.vram_stats()
        free = stats["free"] or 0
        usable = free * (1.0 - self.reserve_fraction)
        bytes_per_token = hidden_size * max(1, n_layers) * 10
        approx_per_sample = max(1, seq_len) * bytes_per_token
        recommended = int(max(1, usable // max(1, approx_per_sample)))
        return max(1, min(int(requested_batch_size), recommended))

    def recommend_seq_len(self, requested_seq_len, hidden_size, n_layers, batch_size):
        if not self.cuda_available():
            return requested_seq_len
        stats = self.vram_stats()
        free = stats["free"] or 0
        usable = free * (1.0 - self.reserve_fraction)
        bytes_per_token = hidden_size * max(1, n_layers) * max(1, batch_size) * 10
        recommended = int(max(64, usable // max(1, bytes_per_token)))
        return max(64, min(int(requested_seq_len), recommended))

    def empty_cache(self):
        if self.cuda_available():
            torch.cuda.empty_cache()
