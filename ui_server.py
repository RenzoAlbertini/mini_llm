import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


class UIRuntime:
    def __init__(self, checkpoint, quantized_checkpoint, tokenizer_path, finetuned_checkpoint=None):
        self.checkpoint = checkpoint
        self.quantized_checkpoint = quantized_checkpoint
        self.finetuned_checkpoint = finetuned_checkpoint or "models/checkpoints/mini_llm_32m_finetuned.pt"
        self.tokenizer_path = tokenizer_path
        self.loaded = {}
        self.tokenizer = None
        self.device = None
        self.last_tokens_per_second = 0.0
        self.last_latency_ms = 0.0
        self.last_error = None

    @property
    def torch_available(self):
        try:
            import torch  # noqa: F401

            return True
        except ModuleNotFoundError:
            return False

    def load(self, model_type="32m"):
        if model_type == "demo":
            return None
        if model_type in self.loaded:
            return self.loaded[model_type]

        try:
            from inference.generate import load_model
            from tokenizer.tokenizer import BPETokenizer
            from utils.helpers import get_device

            self.device = get_device()
            self.tokenizer = self.tokenizer or BPETokenizer.load_model(self.tokenizer_path)
            quantized = model_type in {"32m-4bit", "8bit"}
            if model_type == "finetuned":
                checkpoint = self.finetuned_checkpoint
            elif quantized and self.quantized_checkpoint:
                checkpoint = self.quantized_checkpoint
            else:
                checkpoint = self.checkpoint
            model = load_model(checkpoint, self.device, quantized=quantized)

            if model_type == "fp16" and self.device.type == "cuda":
                model = model.half()
            elif model_type == "bf16" and self.device.type == "cuda":
                import torch

                model = model.to(dtype=torch.bfloat16)

            self.loaded[model_type] = model
            self.last_error = None
            return model
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def available_models(self):
        return [
            {
                "id": "32m",
                "label": "MiniLLM-32M",
                "available": Path(self.checkpoint).exists() and self.torch_available,
            },
            {
                "id": "32m-4bit",
                "label": "MiniLLM-32M 4-bit",
                "available": bool(self.quantized_checkpoint) and Path(self.quantized_checkpoint).exists() and self.torch_available,
            },
            {
                "id": "finetuned",
                "label": "Finetuned",
                "available": Path(self.finetuned_checkpoint).exists() and self.torch_available,
            },
            {
                "id": "demo",
                "label": "Demo fallback",
                "available": True,
            },
        ]

    def metrics(self):
        ram = memory_info()
        vram = {"available": False, "allocated": None, "reserved": None, "total": None, "free": None, "used_percent": None}
        if self.torch_available:
            try:
                from memory_manager import MemoryManager

                vram = MemoryManager().vram_stats()
            except Exception:
                pass
        return {
            "tokens_per_second": self.last_tokens_per_second,
            "latency_ms": self.last_latency_ms,
            "ram": ram,
            "vram": vram,
            "last_error": self.last_error,
        }


def memory_info():
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return {
                "available": True,
                "total": int(status.ullTotalPhys),
                "free": int(status.ullAvailPhys),
                "used_percent": int(status.dwMemoryLoad),
            }
        except Exception:
            pass
    return {"available": False, "total": None, "free": None, "used_percent": None}


def demo_tokens(prompt, max_new_tokens):
    base = (
        prompt.strip()
        + " -> mini_llm UI is running in demo mode. Train or load a checkpoint to use real model output."
    )
    words = base.split()
    if not words:
        words = ["mini_llm", "demo", "output"]
    for i in range(max_new_tokens):
        yield words[i % len(words)] + " "


async def stream_real(runtime, websocket, payload):
    model_type = payload.get("model", "32m")
    model = runtime.load(model_type)
    if model is None and runtime.quantized_checkpoint and Path(runtime.quantized_checkpoint).exists():
        await websocket.send_json({"type": "status", "message": "fallback to 4-bit model"})
        model_type = "32m-4bit"
        model = runtime.load("32m-4bit")

    prompt = payload.get("prompt", "")
    max_new_tokens = int(payload.get("max_new_tokens", 120))
    start = time.perf_counter()
    count = 0

    if model is None:
        await websocket.send_json({"type": "status", "message": "demo fallback active"})
        for token in demo_tokens(prompt, max_new_tokens):
            await websocket.send_json({"type": "token", "text": token})
            count += 1
            await asyncio.sleep(0.035)
    else:
        import torch

        runtime.tokenizer = runtime.tokenizer or __import__("tokenizer.tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load_model(runtime.tokenizer_path)
        ids = runtime.tokenizer.encode(prompt, add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=runtime.device)
        model.eval()
        temperature = float(payload.get("temperature", 0.8))
        top_k = int(payload.get("top_k", 50))
        top_p = float(payload.get("top_p", 0.95))
        from inference.generate import top_k_filter, top_p_filter
        import torch.nn.functional as F

        dynamic_context = bool(payload.get("dynamic_context", True))
        max_context = int(payload.get("max_context", model.config.seq_len))

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context_limit = min(model.config.seq_len, max_context) if dynamic_context else model.config.seq_len
                x_cond = x[:, -context_limit:]
                logits, _ = model(x_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                logits = top_k_filter(logits, top_k)
                logits = top_p_filter(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_id], dim=1)
                token = runtime.tokenizer.decode([next_id.item()])
                await websocket.send_json({"type": "token", "text": token})
                count += 1
                if count % 8 == 0:
                    elapsed_now = max(1e-6, time.perf_counter() - start)
                    runtime.last_tokens_per_second = count / elapsed_now
                    runtime.last_latency_ms = elapsed_now / max(1, count) * 1000.0
                    await websocket.send_json({"type": "metrics", "metrics": runtime.metrics()})
                if next_id.item() == runtime.tokenizer.eos_id:
                    break

    elapsed = max(1e-6, time.perf_counter() - start)
    runtime.last_tokens_per_second = count / elapsed
    runtime.last_latency_ms = elapsed / max(1, count) * 1000.0
    await websocket.send_json(
        {
            "type": "done",
            "tokens": count,
            "seconds": elapsed,
            "tokens_per_second": runtime.last_tokens_per_second,
        }
    )


def create_app(runtime):
    app = FastAPI(title="mini_llm UI")
    app.mount("/static", StaticFiles(directory="ui"), name="static")

    @app.get("/ui")
    def ui():
        return FileResponse("ui/index.html")

    @app.get("/")
    def root():
        return FileResponse("ui/index.html")

    @app.get("/metrics")
    def metrics():
        return runtime.metrics()

    @app.get("/models")
    def models():
        return {"models": runtime.available_models(), "active_error": runtime.last_error}

    @app.get("/agents")
    def agents():
        return {"agents": discover_optional_items([".agents", ".codex/agents"])}

    @app.get("/plugins")
    def plugins():
        return {"plugins": discover_optional_items([".codex/plugins", ".agents/plugins"])}

    @app.post("/generate")
    async def generate_endpoint(payload: dict):
        chunks = []

        class Collector:
            async def send_json(self, item):
                if item.get("type") == "token":
                    chunks.append(item["text"])

        await stream_real(runtime, Collector(), payload)
        return {"text": "".join(chunks), "metrics": runtime.metrics()}

    @app.websocket("/stream")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        try:
            payload = await websocket.receive_json()
            await stream_real(runtime, websocket, payload)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            runtime.last_error = f"{type(exc).__name__}: {exc}"
            await websocket.send_json({"type": "error", "message": runtime.last_error})

    return app


def discover_optional_items(paths):
    items = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        for child in sorted(path.iterdir()):
            items.append({"name": child.name, "path": str(child), "type": "dir" if child.is_dir() else "file"})
    return items


DEFAULT_RUNTIME = UIRuntime(
    checkpoint=os.environ.get("MINI_LLM_CHECKPOINT", "models/checkpoints/mini_llm_32m_best.pt"),
    quantized_checkpoint=os.environ.get("MINI_LLM_QUANTIZED_CHECKPOINT", "models/quantized/mini_llm_32m_4bit.pt"),
    tokenizer_path=os.environ.get("MINI_LLM_TOKENIZER", "tokenizer/tokenizer.json"),
    finetuned_checkpoint=os.environ.get("MINI_LLM_FINETUNED_CHECKPOINT", "models/checkpoints/mini_llm_32m_finetuned.pt"),
)
app = create_app(DEFAULT_RUNTIME)


def main():
    parser = argparse.ArgumentParser(description="Avvia la UI web interattiva mini_llm.")
    parser.add_argument("--checkpoint", default="models/checkpoints/mini_llm_32m_best.pt")
    parser.add_argument("--quantized_checkpoint", default="models/quantized/mini_llm_32m_4bit.pt")
    parser.add_argument("--finetuned_checkpoint", default="models/checkpoints/mini_llm_32m_finetuned.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    runtime = UIRuntime(args.checkpoint, args.quantized_checkpoint, args.tokenizer, args.finetuned_checkpoint)
    local_app = create_app(runtime)
    print(f"mini_llm UI: http://{args.host}:{args.port}/ui")
    uvicorn.run(local_app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
