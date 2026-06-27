import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Runtime:
    def __init__(self, checkpoint, tokenizer_path, quantized=False, quantized_checkpoint=None):
        self.checkpoint = checkpoint
        self.tokenizer_path = tokenizer_path
        self.quantized = quantized
        self.quantized_checkpoint = quantized_checkpoint
        self.device = None
        self.tokenizer = None
        self.model = None
        self.error = None

    def load(self):
        if self.model is not None or self.error is not None:
            return
        try:
            from inference.generate import load_model
            from tokenizer.tokenizer import BPETokenizer
            from utils.helpers import count_parameters, get_device

            self.device = get_device()
            self.tokenizer = BPETokenizer.load_model(self.tokenizer_path)
            self.model = load_model(self.checkpoint, self.device, quantized=self.quantized)
            self.parameters = count_parameters(self.model)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.quantized_checkpoint:
                from inference.generate import load_model
                from tokenizer.tokenizer import BPETokenizer
                from utils.helpers import count_parameters, get_device

                self.device = get_device()
                self.tokenizer = BPETokenizer.load_model(self.tokenizer_path)
                self.model = load_model(self.quantized_checkpoint, self.device, quantized=True)
                self.quantized = True
                self.parameters = count_parameters(self.model)
            else:
                self.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"


def run_with_timeout(fn, timeout):
    box = {"ok": False, "value": None, "error": None}

    def target():
        try:
            box["value"] = fn()
            box["ok"] = True
        except Exception as exc:
            box["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return False, None, "timeout"
    return box["ok"], box["value"], box["error"]


def make_handler(runtime, production=False, timeout=30.0):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload, status=200):
            raw = json.dumps(payload, indent=None if production else 2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):
            runtime.load()
            if self.path == "/health":
                self._send({"ok": runtime.error is None, "error": runtime.error})
            elif self.path == "/info":
                config = runtime.model.config.to_dict() if runtime.model is not None else None
                self._send({
                    "checkpoint": runtime.checkpoint,
                    "tokenizer": runtime.tokenizer_path,
                    "quantized": runtime.quantized,
                    "device": str(runtime.device),
                    "parameters": getattr(runtime, "parameters", None),
                    "config": config,
                    "error": runtime.error,
                })
            else:
                self._send({"error": "not found"}, status=404)

        def do_POST(self):
            runtime.load()
            if runtime.error:
                self._send({"error": runtime.error}, status=500)
                return
            body = self._body()
            if self.path == "/generate":
                def work():
                    from inference.generate import generate

                    return generate(
                        runtime.model,
                        runtime.tokenizer,
                        body.get("prompt", "python is"),
                        max_new_tokens=int(body.get("max_new_tokens", 80)),
                        temperature=float(body.get("temperature", 0.8)),
                        top_k=int(body.get("top_k", 50)),
                        top_p=float(body.get("top_p", 0.95)),
                        device=runtime.device,
                    )

                ok, value, error = run_with_timeout(work, timeout)
                self._send({"text": value} if ok else {"error": error}, status=200 if ok else 500)
            elif self.path == "/evaluate":
                def work_eval():
                    import math
                    import torch

                    text = body.get("text", body.get("prompt", "python is a model"))
                    ids = runtime.tokenizer.encode(text, add_bos=True, add_eos=True)
                    if len(ids) < 2:
                        return {"loss": None, "perplexity": None, "tokens": len(ids)}
                    ids = ids[-runtime.model.config.seq_len - 1:]
                    x = torch.tensor([ids[:-1]], dtype=torch.long, device=runtime.device)
                    y = torch.tensor([ids[1:]], dtype=torch.long, device=runtime.device)
                    with torch.no_grad():
                        _, loss = runtime.model(x, y)
                    loss_value = float(loss.item())
                    return {"loss": loss_value, "perplexity": math.exp(min(loss_value, 20.0)), "tokens": len(ids)}

                ok, value, error = run_with_timeout(work_eval, timeout)
                self._send(value if ok else {"error": error}, status=200 if ok else 500)
            else:
                self._send({"error": "not found"}, status=404)

        def log_message(self, fmt, *args):
            if not production:
                super().log_message(fmt, *args)

    return Handler


def create_fastapi_app(runtime, timeout=30.0):
    try:
        from fastapi import FastAPI
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI non installato. Usa il server standard o installa fastapi.") from exc

    app = FastAPI(title="mini_llm API")

    @app.get("/health")
    def health():
        runtime.load()
        return {"ok": runtime.error is None, "error": runtime.error}

    @app.get("/info")
    def info():
        runtime.load()
        return {
            "checkpoint": runtime.checkpoint,
            "tokenizer": runtime.tokenizer_path,
            "quantized": runtime.quantized,
            "device": str(runtime.device),
            "parameters": getattr(runtime, "parameters", None),
            "config": runtime.model.config.to_dict() if runtime.model is not None else None,
            "error": runtime.error,
        }

    @app.post("/generate")
    def generate_endpoint(payload: dict):
        runtime.load()
        if runtime.error:
            return {"error": runtime.error}
        from inference.generate import generate

        text = generate(
            runtime.model,
            runtime.tokenizer,
            payload.get("prompt", "python is"),
            max_new_tokens=int(payload.get("max_new_tokens", 80)),
            temperature=float(payload.get("temperature", 0.8)),
            top_k=int(payload.get("top_k", 50)),
            top_p=float(payload.get("top_p", 0.95)),
            device=runtime.device,
        )
        return {"text": text}

    @app.post("/evaluate")
    def evaluate_endpoint(payload: dict):
        runtime.load()
        if runtime.error:
            return {"error": runtime.error}
        import math
        import torch

        ids = runtime.tokenizer.encode(payload.get("text", "python is a model"), add_bos=True, add_eos=True)
        ids = ids[-runtime.model.config.seq_len - 1:]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=runtime.device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=runtime.device)
        with torch.no_grad():
            _, loss = runtime.model(x, y)
        loss_value = float(loss.item())
        return {"loss": loss_value, "perplexity": math.exp(min(loss_value, 20.0))}

    return app


def main():
    parser = argparse.ArgumentParser(description="API locale mini_llm.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--quantized_checkpoint", default=None)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["debug", "production"], default="debug")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    runtime = Runtime(args.checkpoint, args.tokenizer, args.quantized, args.quantized_checkpoint)

    try:
        import fastapi  # noqa: F401
        print("FastAPI rilevato, ma avvio il server standard library per restare zero-dipendenza runtime.")
    except Exception:
        pass

    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime, args.mode == "production", args.timeout))
    print(f"mini_llm API su http://{args.host}:{args.port}")
    print("endpoint: GET /health, GET /info, POST /generate, POST /evaluate")
    server.serve_forever()


if __name__ == "__main__":
    main()
