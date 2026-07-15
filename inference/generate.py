import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.transformer import MiniTransformerLM
from tokenizer.tokenizer import BPETokenizer
from utils.helpers import get_device, load_checkpoint, set_seed
from utils.quantize_4bit import load_4bit_model
from utils.quantization import load_quantized_model


DEFAULT_CHECKPOINT_CANDIDATES = (
    "models/checkpoints/best.pt",
    "models/checkpoints/mini_llm_32m_best.pt",
    "models/checkpoints/demo/best.pt",
    "models/checkpoints/final.pt",
)


def resolve_checkpoint_path(checkpoint_path=None):
    """Resolve an explicit checkpoint or the first conventional local checkpoint."""
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint non trovato: {path}")
        return str(path)

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    searched = ", ".join(DEFAULT_CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        "Nessun checkpoint disponibile. Esegui prima il Quick Start demo oppure "
        f"passa --checkpoint. Percorsi controllati: {searched}"
    )


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

    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits.masked_fill(remove, float("-inf")),
    )
    return filtered


def parse_stop_sequences(tokenizer, stop_texts):
    stop_sequences = [(tokenizer.eos_id,)]
    for text in stop_texts or []:
        ids = tokenizer.encode(text)
        if ids:
            stop_sequences.append(tuple(ids))
    return stop_sequences


def matching_stop_sequence(ids, stop_sequences):
    for stop in stop_sequences:
        if len(ids) >= len(stop) and tuple(ids[-len(stop):]) == stop:
            return stop
    return None


def ends_with_stop_sequence(ids, stop_sequences):
    return matching_stop_sequence(ids, stop_sequences) is not None


def apply_repetition_penalty(logits, generated_ids, penalty=1.0):
    """Apply the standard sign-aware repetition penalty to generated tokens only."""
    if not penalty or penalty <= 1.0:
        return logits
    for token_id in set(generated_ids):
        score = logits[:, token_id]
        logits[:, token_id] = torch.where(score < 0, score * penalty, score / penalty)
    return logits


def apply_guidance(
    logits,
    generated_ids,
    tokenizer,
    required_words=None,
    bad_words=None,
    repetition_penalty=1.0,
    bad_token_penalty=0.0,
):
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if bad_token_penalty > 0:
        for word in bad_words or []:
            for token_id in tokenizer.encode(word):
                logits[:, token_id] = logits[:, token_id] - bad_token_penalty

    text_so_far = tokenizer.decode(generated_ids).casefold()
    missing = [word for word in (required_words or []) if word.casefold() not in text_so_far]
    if missing:
        ids = tokenizer.encode(missing[0])
        if ids:
            logits[:, ids[0]] = logits[:, ids[0]] + 1.0
    return logits


def score_sample(text, required_words=None):
    folded = text.casefold()
    score = sum(10 for word in required_words or [] if word.casefold() in folded)
    words = folded.split()
    score -= max(0, len(words) - len(set(words)))
    return score


def validate_generation_args(max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens deve essere almeno 1")
    if float(temperature) < 0:
        raise ValueError("temperature non puo essere negativa")
    if top_k is not None and int(top_k) < 0:
        raise ValueError("top_k non puo essere negativo")
    if top_p is not None and not 0 < float(top_p) <= 1:
        raise ValueError("top_p deve essere compreso tra 0 e 1")
    if float(repetition_penalty) < 1:
        raise ValueError("repetition_penalty deve essere almeno 1")


def _make_generator(device, seed):
    if seed is None or device.type not in {"cpu", "cuda"}:
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _next_token(logits, do_sample, generator=None):
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = F.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or torch.any(probs.sum(dim=-1) <= 0):
        raise RuntimeError("Distribuzione di probabilita non valida durante la generazione")
    return torch.multinomial(probs, num_samples=1, generator=generator)


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
    repetition_penalty=1.08,
    bad_token_penalty=0.0,
    do_sample=False,
    seed=42,
    return_full_text=False,
    device=None,
):
    validate_generation_args(max_new_tokens, temperature, top_k, top_p, repetition_penalty)
    model.eval()
    device = device or next(model.parameters()).device
    stop_sequences = stop_sequences or [(tokenizer.eos_id,)]
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = []
    streamed_text = ""
    generator = _make_generator(device, seed)

    for _ in range(int(max_new_tokens)):
        x_cond = x[:, -model.config.seq_len :]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]
        if do_sample:
            logits = logits / max(float(temperature), 1e-6)
        logits = apply_guidance(
            logits,
            generated_ids,
            tokenizer,
            required_words=required_words,
            bad_words=bad_words,
            repetition_penalty=repetition_penalty,
            bad_token_penalty=bad_token_penalty,
        )
        if do_sample:
            logits = top_k_filter(logits, top_k)
            logits = top_p_filter(logits, top_p)
        next_id = _next_token(logits, do_sample=do_sample, generator=generator)
        token_id = int(next_id.item())
        generated_ids.append(token_id)

        stop = matching_stop_sequence(generated_ids, stop_sequences)
        if stop is not None:
            del generated_ids[-len(stop) :]
            break

        x = torch.cat([x, next_id], dim=1)
        if stream_callback is not None:
            decoded = tokenizer.decode(generated_ids)
            if not decoded.endswith("\ufffd"):
                piece = decoded[len(streamed_text) :] if decoded.startswith(streamed_text) else decoded
                if piece:
                    stream_callback(piece)
                streamed_text = decoded

    completion = tokenizer.decode(generated_ids)
    if stream_callback is not None and len(completion) > len(streamed_text):
        stream_callback(completion[len(streamed_text) :])
    if return_full_text:
        return prompt + completion
    return completion


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
    repetition_penalty=1.08,
    bad_token_penalty=0.0,
    num_samples=1,
    do_sample=False,
    seed=42,
    return_full_text=False,
    device=None,
):
    samples = []
    for sample_idx in range(max(1, int(num_samples))):
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
            do_sample=do_sample,
            seed=None if seed is None else int(seed) + sample_idx,
            return_full_text=return_full_text,
            device=device,
        )
        samples.append(text)
    return max(samples, key=lambda text: score_sample(text, required_words=required_words))


def _validate_checkpoint(checkpoint, checkpoint_path):
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint non valido (payload non strutturato): {checkpoint_path}")
    missing = {"config", "model"} - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint non valido, campi mancanti {sorted(missing)}: {checkpoint_path}")


def validate_model_tokenizer(model, tokenizer):
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer e checkpoint incompatibili: "
            f"vocab tokenizer={tokenizer.vocab_size}, vocab modello={model.config.vocab_size}"
        )


def load_model(checkpoint_path, device, quantized=False):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if quantized:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_checkpoint(checkpoint, checkpoint_path)
        config = ModelConfig.from_dict(checkpoint["config"])
        model = MiniTransformerLM(config).to(device)
        if checkpoint.get("quantization") == "4bit":
            model, _ = load_4bit_model(
                checkpoint_path,
                model,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device=device,
            )
            return model.eval()
        model, _ = load_quantized_model(checkpoint_path, model, device=device)
        return model.eval()

    checkpoint = load_checkpoint(checkpoint_path, device=device)
    _validate_checkpoint(checkpoint, checkpoint_path)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = MiniTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def generate_from_prompt(
    prompt,
    checkpoint_path=None,
    tokenizer_path="tokenizer/tokenizer.json",
    max_new_tokens=120,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    quantized=False,
    stream_callback=None,
    required_words=None,
    bad_words=None,
    repetition_penalty=1.08,
    bad_token_penalty=0.0,
    num_samples=1,
    do_sample=False,
    seed=42,
):
    device = get_device()
    tokenizer = BPETokenizer.load_model(tokenizer_path)
    model = load_model(checkpoint_path, device, quantized=quantized)
    validate_model_tokenizer(model, tokenizer)
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
        do_sample=do_sample,
        seed=seed,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Genera una continuazione da un checkpoint MiniLLM.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", default="C'era una volta")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--sample", action="store_true", help="Abilita campionamento; il default e greedy e deterministico.")
    parser.add_argument("--stop", action="append", default=[], help="Sequenza di stop testuale. Ripetibile.")
    parser.add_argument("--required_word", action="append", default=[])
    parser.add_argument("--bad_word", action="append", default=[])
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--bad_token_penalty", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--quantized", action="store_true", help="Carica un checkpoint quantizzato.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    model = load_model(args.checkpoint, device, quantized=args.quantized)
    validate_model_tokenizer(model, tokenizer)
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
        do_sample=args.sample,
        seed=args.seed,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
