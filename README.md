# MiniLLM — From-Scratch Local Language Model

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Tests](https://github.com/RenzoAlbertini/mini_llm/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

MiniLLM is a compact decoder-only Transformer implemented with PyTorch. It
covers the complete educational pipeline: byte-level BPE tokenization,
training, validation, checkpointing, deterministic or sampled inference,
benchmarking, and a model-first local chat interface.

This is a learning and systems-engineering project, not a pretrained assistant.
No trained checkpoint is committed to the repository. Output quality therefore
depends on the corpus, training time, and checkpoint that you create locally.

## What is implemented

- A GPT-style, decoder-only Transformer with causal self-attention.
- A byte-level BPE tokenizer that can represent arbitrary UTF-8 text.
- Leakage-resistant contiguous train/validation splits.
- Fingerprinted token caches that invalidate when data or tokenizer changes.
- Checkpoint resume, early stopping, mixed precision, gradient checkpointing,
  and GPU thermal controls.
- Greedy deterministic decoding by default, with opt-in seeded sampling.
- Sign-correct repetition penalty, top-k/top-p filtering, and stop sequences.
- A Chat Mode that runs the selected checkpoint for every request.
- Explicit response provenance: `model` or `quality_fallback_after_model`.
- Benchmark, dashboard, API, export, and quantization experiments.
- CPU-compatible automated tests in GitHub Actions.

## Architecture

The `mini_llm_32m` compatibility preset is approximately 36 million parameters:

| Setting | Value |
|---|---:|
| Vocabulary | 8,192 |
| Context length | 512 |
| Transformer layers | 12 |
| Attention heads | 8 |
| Hidden size | 512 |
| Feed-forward size | 1,536 |

The historical preset name is retained for CLI compatibility even though the
actual parameter count is closer to 36M.

## Quick Start — verified smoke test

The Quick Start intentionally trains a tiny model for 20 steps. Its purpose is
to verify the complete pipeline in a few minutes, not to produce an assistant-
quality checkpoint.

```bash
git clone https://github.com/RenzoAlbertini/mini_llm.git
cd mini_llm
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install and run the smoke training:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_training.py --demo --max_steps 20
```

Generate a deterministic continuation from the resulting checkpoint:

```bash
python run_generate.py \
  --checkpoint models/checkpoints/demo/best.pt \
  --tokenizer data/processed/demo_tokenizer.json \
  --prompt "python is" \
  --max_new_tokens 40
```

On PowerShell, place the command on one line or replace `\` with the PowerShell
continuation character.

## Deterministic and sampled generation

Generation is greedy and repeatable by default. This makes checkpoint
comparisons meaningful and prevents the CLI from appearing random:

```bash
python run_generate.py --checkpoint PATH --tokenizer PATH --prompt "python is"
```

Sampling is explicit and reproducible with a seed:

```bash
python run_generate.py --checkpoint PATH --tokenizer PATH --prompt "python is" --sample --seed 42 --temperature 0.8 --top_p 0.9
```

The generator returns only newly generated text; it no longer repeats the
prompt in the result.

## Chat Mode

Start Chat Mode with the demo checkpoint:

```bash
python chat/server.py \
  --checkpoint models/checkpoints/demo/best.pt \
  --tokenizer data/processed/demo_tokenizer.json \
  --checkpoint_dir models/checkpoints \
  --port 8020
```

Open `http://127.0.0.1:8020/chat`.

Every request first executes the selected model. If the generated text fails a
basic coherence check, the API may return a conservative fallback and labels it
`quality_fallback_after_model`; it never silently presents a template as model
output. Missing or incompatible checkpoints produce a clear HTTP error.

API endpoints:

```http
POST /api/chat
GET /api/chat/checkpoints
```

Example non-streaming request:

```bash
curl -X POST http://127.0.0.1:8020/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Spiega un tokenizer.","max_tokens":80,"do_sample":false,"history":[]}'
```

## Training a meaningful experiment

The bundled JSON data are deterministic synthetic fixtures for smoke testing;
they are not copies of Wikipedia, Project Gutenberg, SQuAD, or OpenAssistant.
See [`data/raw/README.md`](data/raw/README.md) for provenance.

For a serious experiment, supply a larger, diverse corpus that you are licensed
to use. The builder can still prepare the bundled fixtures without the old
hard-coded 25 MB failure condition:

```bash
python data/raw/build_dataset.py
python tokenizer/build_tokenizer.py --dataset data/processed/train.txt --out tokenizer/tokenizer.json --vocab_size 8192
```

Example laptop-oriented 36M training command:

```bash
python run_training.py --model_size mini_llm_32m --seq_len 256 --batch_size 1 --epochs 8 --gradient_checkpointing --fp16 --gpu_memory_fraction 0.70 --gpu_max_temp 80 --thermal_cooldown_seconds 10 --eval_every 200 --eval_batches 5 --checkpoint_dir models/checkpoints
```

This is a long training run, not part of the Quick Start.

## Dashboard

```bash
python dashboard.py --port 8010
```

Open `http://127.0.0.1:8010`. The dashboard displays training loss, GPU
temperature and utilization, VRAM, checkpoints, controls, and benchmark output.

![MiniLLM training dashboard](benchmark/Screensht.png)

## Evaluation

```bash
python evaluate.py --checkpoint models/checkpoints/best.pt
```

Results are saved under `data/benchmarks/`. Perplexity, token accuracy,
coherence heuristics, and repetition scores are intended for comparisons
between this project's checkpoints. They are not directly comparable with
instruction-tuned production LLM benchmarks.

## Tests

```bash
python run_all_tests.py
python verify_project_structure.py
```

The runner treats skipped dependency-backed tests as failures, so an `OK`
summary now means that the model, generation, dataset, Chat Mode, Quick Start,
and end-to-end paths actually ran.

## Project layout

```text
model/       Transformer architecture and configurations
training/    dataset, trainer, checkpoints, controls, and statistics
tokenizer/   byte-level BPE implementation and builder
inference/   shared deterministic/sampled generation path
chat/        model-first local Chat Mode
benchmark/   evaluation dataset and metrics
data/        educational fixtures and generated local artifacts
ui/          separate inference UI assets
tests/       automated regression suite
```

## Limitations

- A from-scratch 36M model needs substantially more high-quality data and
  compute than the smoke demo to produce coherent language.
- The bundled corpus is synthetic and deliberately small.
- Int8/int4 utilities compress checkpoints and then dequantize weights for the
  standard PyTorch model; they are not optimized quantized inference kernels.
- The heuristic quality fallback is transparent but does not improve the model
  weights. Better quality requires better data, training, and evaluation.

## License

MIT License. See [`LICENSE`](LICENSE).
