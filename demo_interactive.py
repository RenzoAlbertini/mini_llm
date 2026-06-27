import argparse


def main():
    parser = argparse.ArgumentParser(description="Chat interattiva terminale con mini_llm.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--stream", action="store_true", help="Compatibilita CLI: la demo stampa gia in streaming.")
    args = parser.parse_args()

    try:
        from inference.generate import generate, load_model
        from tokenizer.tokenizer import BPETokenizer
        from utils.helpers import get_device
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        return 1

    device = get_device()
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    model = load_model(args.checkpoint, device, quantized=args.quantized)

    print("mini_llm interactive demo. Scrivi /exit per uscire.")
    while True:
        prompt = input("\nTu> ").strip()
        if prompt in {"/exit", "/quit"}:
            break
        if not prompt:
            continue
        print("MiniLLM> ", end="", flush=True)

        def stream(piece):
            print(piece, end="", flush=True)

        generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stream_callback=stream,
            device=device,
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
