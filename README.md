# MiniLLM - Lightweight Local Language Model

![Python](benchmark/Screenshot.png)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

MiniLLM is a compact local language-model project built with PyTorch: dataset builder, byte-level tokenizer, MiniLLM-36M training, dashboard, benchmark suite, and a local Chat Mode.

It is designed for learning, experimentation, and laptop-friendly training on consumer GPUs.

![MiniLLM Web UI](ui/screenshot.png)

## Features

- MiniLLM-36M decoder-only transformer (the legacy CLI preset is named `mini_llm_32m`).
- Byte-level BPE tokenizer with Italian, punctuation, UTF-8, and code support.
- Dataset builder for natural text and small synthetic QA, dialogue, instruction-tuning, and technical examples.
- Laptop-safe training profile with FP16, gradient checkpointing, checkpoint resume, and temperature cooldown.
- Local FastAPI dashboard with loss charts, GPU temperature, utilization, VRAM, logs, checkpoints, and benchmark plots.
- Benchmark Suite with perplexity, average log-likelihood, token accuracy, coherence, and repetition metrics.
- Chat Mode with streaming responses, checkpoint selector, history, and professional fallback guardrails.
- Export and checkpoint-compression helpers using portable int8/int4 quantization experiments.

## Scope and limitations

MiniLLM is a from-scratch educational model, not an attempt to compete with
pretrained production LLMs. The default corpus is deliberately small and contains
synthetic, template-generated examples inspired by QA and assistant datasets; it
does not bundle the official Wikipedia, Gutenberg, SQuAD, or OpenAssistant
datasets. The project is intended to demonstrate the complete language-model
pipeline on consumer hardware.

Training and validation are split on the token stream before sliding windows are
created, with a context-length gap between them. This avoids the severe leakage
that occurs when overlapping windows are randomly divided after tokenization.

The int8/int4 utilities compress checkpoints, then dequantize weights when loading
the standard PyTorch model. They reduce file size but do not provide true int8/NF4
inference kernels or guaranteed speed/VRAM improvements.

## Quick Start

```bash
git clone https://github.com/RenzoAlbertini/mini_llm.git
cd mini_llm
python -m pip install -r requirements.txt
```

Build dataset and tokenizer:

```bash
python data/raw/build_dataset.py --force
python tokenizer/build_tokenizer.py --dataset data/processed/train.txt --out tokenizer/tokenizer.json --vocab_size 8192
```

Start laptop-safe training:

```bash
python run_training.py --model_size mini_llm_32m --seq_len 256 --batch_size 1 --epochs 8 --gradient_checkpointing --fp16 --gpu_memory_fraction 0.70 --gpu_max_temp 80 --thermal_cooldown_seconds 10 --eval_every 200 --eval_batches 5 --checkpoint_dir models/checkpoints
```

Training writes:

```text
models/checkpoints/best.pt
models/checkpoints/last.pt
models/checkpoints/final.pt
data/logs/training.log
data/logs/training_stats.csv
```

## Dashboard

```bash
python dashboard.py --port 8010
```

Open `http://127.0.0.1:8010`.

The dashboard monitors training in real time and includes plots, GPU stats, checkpoint status, benchmark results, and training controls.

## Chat Mode

```bash
python chat/server.py --checkpoint models/checkpoints/best.pt --port 8020
```

Open `http://127.0.0.1:8020/chat`.

API endpoints:

```http
POST /api/chat
GET /api/chat/checkpoints
```

Example:

```bash
curl -X POST http://127.0.0.1:8020/api/chat -H "Content-Type: application/json" -d "{\"prompt\":\"Ciao, come ti chiami?\",\"temperature\":0.45,\"top_p\":0.82,\"max_tokens\":80,\"history\":[]}"
```

## Benchmark

```bash
python evaluate.py --checkpoint models/checkpoints/best.pt
```

Results are saved in `data/benchmarks/results_<checkpoint>.json`.

Dashboard API:

```http
POST /api/evaluate
Content-Type: application/json

{"checkpoint_path":"models/checkpoints/best.pt"}
```

## Project Layout

```text
model/       transformer architecture and configs
training/    trainer, dataset loader, checkpoints, controls
tokenizer/   BPE tokenizer build and artifacts
data/        raw builders, processed dataset, logs, plots
chat/        local Chat Mode API and UI
benchmark/   evaluation dataset and metrics
ui/          local UI assets
utils/       helpers, quantization, plotting
tests/       test suite
export/      export artifacts
```

## Release

Current release: `v1.0.0`

Roadmap:

- Larger instruction-tuning corpus.
- LoRA fine-tuning.
- More benchmark categories.
- Model card and dataset card.
- ONNX or TorchScript export.

## License

MIT License. See [LICENSE](LICENSE).
