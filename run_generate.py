import argparse
import sys
import threading
import time

from inference.generate import generate, load_model, parse_stop_sequences
from tokenizer.tokenizer import BPETokenizer
from utils.helpers import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Genera testo da un checkpoint mini_llm.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--quantized_checkpoint", default=None)
    parser.add_argument("--mode", choices=["debug", "standard", "production"], default="standard")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", default="python is")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--required_word", action="append", default=[])
    parser.add_argument("--bad_word", action="append", default=[])
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--bad_token_penalty", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    try:
        model = load_model(args.checkpoint, device, quantized=args.quantized)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and args.quantized_checkpoint:
            if args.mode != "production":
                print("OOM rilevato: fallback a checkpoint quantizzato")
            model = load_model(args.quantized_checkpoint, device, quantized=True)
            args.quantized = True
        else:
            raise
    stop_sequences = parse_stop_sequences(tokenizer, args.stop)
    model_type = "quantizzato 8-bit" if args.quantized else "normale"

    if args.mode != "production":
        print(f"checkpoint: {args.checkpoint}")
        print(f"modello: {model_type}")
        print(f"prompt: {args.prompt}")
        if args.mode == "debug":
            ids = tokenizer.encode(args.prompt, add_bos=True)
            print(f"prompt_token_ids: {ids}")
            print(f"device: {device}")
        print("output:")

    def stream_piece(piece):
        print(piece, end="", flush=True)

    result = {"text": None, "error": None}

    def work():
        try:
            result["text"] = generate(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_sequences=stop_sequences,
                stream_callback=stream_piece if args.stream else None,
                required_words=args.required_word,
                bad_words=args.bad_word,
                repetition_penalty=args.repetition_penalty,
                bad_token_penalty=args.bad_token_penalty,
                num_samples=args.num_samples,
                device=device,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and args.quantized_checkpoint:
                fallback = load_model(args.quantized_checkpoint, device, quantized=True)
                result["text"] = generate(
                    fallback,
                    tokenizer,
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device,
                )
            else:
                result["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

    started = time.perf_counter()
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(args.timeout)
    elapsed = time.perf_counter() - started
    if thread.is_alive():
        print("errore: timeout generazione")
        return 1
    if result["error"]:
        print(f"errore: {result['error']}")
        return 1
    text = result["text"]
    if args.stream:
        print()
    else:
        print(text)
    if args.mode != "production":
        print(f"tempo generazione: {elapsed:.2f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
