import traceback


def main():
    print("mini_llm sanity check")
    print("=====================")
    try:
        import torch

        from inference.generate import generate
        from model.config import default_small
        from model.transformer import MiniTransformerLM
        from tokenizer.tokenizer import BPETokenizer
        from training.dataset import TokenDataset
        from training.train import evaluate

        tokenizer = BPETokenizer()
        config = default_small()
        config.vocab_size = tokenizer.vocab_size
        config.seq_len = 16
        config.d_model = 32
        config.n_layers = 2
        config.n_heads = 4
        config.d_ff = 64

        model = MiniTransformerLM(config)
        batch_size = 2
        x = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
        y = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))

        logits, loss = model(x, y)
        print(f"input_ids: {tuple(x.shape)}")
        print(f"targets:   {tuple(y.shape)}")
        print(f"logits:    {tuple(logits.shape)}")
        print(f"loss:      {loss.item():.4f}")

        tokens = torch.randint(0, config.vocab_size, (config.seq_len * 4,))
        dataset = TokenDataset(tokens, seq_len=config.seq_len)
        sample_x, sample_y = dataset[0]
        print(f"dataset x: {tuple(sample_x.shape)}")
        print(f"dataset y: {tuple(sample_y.shape)}")

        text = "ciao mini llm"
        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(encoded)
        print(f"tokenizer ids: {len(encoded)}")
        print(f"round-trip: {decoded == text}")

        generated = generate(
            model,
            tokenizer,
            "ciao",
            max_new_tokens=4,
            top_k=20,
            top_p=0.95,
            device=torch.device("cpu"),
        )
        print(f"generate sample: {generated!r}")

        # Import esplicito usato dal check: evaluate deve essere disponibile.
        assert callable(evaluate)
        print("OK: sanity check completato senza eccezioni")
        return 0
    except ModuleNotFoundError as exc:
        print(f"FAIL: dipendenza mancante: {exc.name}")
        print("Installa PyTorch con: pip install -r requirements.txt")
        return 1
    except Exception:
        print("FAIL: eccezione durante il sanity check")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
