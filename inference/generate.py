import argparse

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.transformer import MiniTransformerLM
from tokenizer.tokenizer import BPETokenizer
from utils.helpers import get_device, load_checkpoint, set_seed
from utils.quantization import load_quantized_model


def top_k_filter(logits, top_k):
    if top_k is None or top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def top_p_filter(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)

    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    filtered = logits.clone()
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits.masked_fill(remove, float("-inf")))
    return filtered


def parse_stop_sequences(tokenizer, stop_texts):
    stop_sequences = [(tokenizer.eos_id,)]
    for text in stop_texts or []:
        ids = tokenizer.encode(text)
        if ids:
            stop_sequences.append(tuple(ids))
    return stop_sequences


def ends_with_stop_sequence(ids, stop_sequences):
    return any(
        len(ids) >= len(stop) and tuple(ids[-len(stop):]) == stop
        for stop in stop_sequences
    )


def apply_guidance(logits, generated_ids, tokenizer, required_words=None, bad_words=None, repetition_penalty=1.0, bad_token_penalty=0.0):
    if repetition_penalty and repetition_penalty > 1.0:
        for token_id in set(generated_ids):
            logits[:, token_id] = logits[:, token_id] / repetition_penalty

    for word in bad_words or []:
        for token_id in tokenizer.encode(word):
            logits[:, token_id] = logits[:, token_id] - bad_token_penalty

    text_so_far = tokenizer.decode(generated_ids)
    missing = [word for word in (required_words or []) if word not in text_so_far]
    if missing:
        # Piccola spinta al primo token della prossima parola obbligatoria.
        ids = tokenizer.encode(missing[0])
        if ids:
            logits[:, ids[0]] = logits[:, ids[0]] + 1.0
    return logits


def score_sample(text, required_words=None):
    score = 0
    for word in required_words or []:
        if word in text:
            score += 10
    words = text.split()
    score -= max(0, len(words) - len(set(words)))
    return score


@torch.no_grad()
def _generate_one(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    stop_sequences=None,
    stream_callback=None,
    required_words=None,
    bad_words=None,
    repetition_penalty=1.0,
    bad_token_penalty=0.0,
    device=None,
):
    model.eval()
    device = device or next(model.parameters()).device
    stop_sequences = stop_sequences or [(tokenizer.eos_id,)]
    ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.seq_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        logits = apply_guidance(
            logits,
            x[0].tolist(),
            tokenizer,
            required_words=required_words,
            bad_words=bad_words,
            repetition_penalty=repetition_penalty,
            bad_token_penalty=bad_token_penalty,
        )
        logits = top_k_filter(logits, top_k)
        logits = top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if stream_callback is not None:
            stream_callback(tokenizer.decode([next_id.item()]))
        if ends_with_stop_sequence(x[0].tolist(), stop_sequences):
            break

    return tokenizer.decode(x[0].tolist())


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    stop_sequences=None,
    stream_callback=None,
    required_words=None,
    bad_words=None,
    repetition_penalty=1.0,
    bad_token_penalty=0.0,
    num_samples=1,
    device=None,
):
    samples = []
    for sample_idx in range(max(1, num_samples)):
        text = _generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_sequences=stop_sequences,
            stream_callback=stream_callback if sample_idx == 0 and num_samples == 1 else None,
            required_words=required_words,
            bad_words=bad_words,
            repetition_penalty=repetition_penalty,
            bad_token_penalty=bad_token_penalty,
            device=device,
        )
        samples.append(text)
    return max(samples, key=lambda text: score_sample(text, required_words=required_words))


def load_model(checkpoint_path, device, quantized=False):
    if quantized:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = ModelConfig.from_dict(checkpoint["config"])
        model = MiniTransformerLM(config).to(device)
        model, _ = load_quantized_model(checkpoint_path, model, device=device)
        return model

    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = MiniTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def generate_from_prompt(
    prompt,
    checkpoint_path="checkpoints/final.pt",
    tokenizer_path="tokenizer/tokenizer.json",
    max_new_tokens=120,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    quantized=False,
    stream_callback=None,
    required_words=None,
    bad_words=None,
    repetition_penalty=1.0,
    bad_token_penalty=0.0,
    num_samples=1,
):
    device = get_device()
    tokenizer = BPETokenizer.load_model(tokenizer_path)
    model = load_model(checkpoint_path, device, quantized=quantized)
    return generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stream_callback=stream_callback,
        required_words=required_words,
        bad_words=bad_words,
        repetition_penalty=repetition_penalty,
        bad_token_penalty=bad_token_penalty,
        num_samples=num_samples,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Genera testo da un checkpoint mini-LLM.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", default="C'era una volta")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--stop", action="append", default=[], help="Sequenza di stop testuale. Ripetibile.")
    parser.add_argument("--required_word", action="append", default=[])
    parser.add_argument("--bad_word", action="append", default=[])
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--bad_token_penalty", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--quantized", action="store_true", help="Carica checkpoint quantizzato 8-bit.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    model = load_model(args.checkpoint, device, quantized=args.quantized)
    stop_sequences = parse_stop_sequences(tokenizer, args.stop)
    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_sequences=stop_sequences,
        required_words=args.required_word,
        bad_words=args.bad_word,
        repetition_penalty=args.repetition_penalty,
        bad_token_penalty=args.bad_token_penalty,
        num_samples=args.num_samples,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
