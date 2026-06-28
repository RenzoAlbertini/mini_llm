# MiniLLM — Lightweight Local Language Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

MiniLLM is an educational local language-model project built with PyTorch. It includes a byte-level BPE tokenizer, a decoder-only MiniLLM-32M transformer, local training, checkpoint resume, benchmark evaluation, a monitoring dashboard, Chat Mode, and export utilities.

The project is designed to run locally on consumer hardware and to stay readable enough for learning, experimentation, and iteration.

## Screenshot

![MiniLLM Web UI](ui/screenshot.png)

## Features

- MiniLLM-32M transformer preset.
- Byte-level BPE tokenizer with `tokenizer.json`, `vocab.json`, and `merges.txt`.
- Dataset builder for natural text, QA, dialogue, instruction tuning, natural responses, and technical text.
- FP16 training, gradient checkpointing, checkpoint resume, and temperature-only GPU cooldown.
- Local dashboard with loss charts, GPU temperature, GPU utilization, VRAM, logs, checkpoints, and benchmark plots.
- Benchmark Suite for perplexity, log-likelihood, token accuracy, coherence, and repetition metrics.
- Chat Mode with FastAPI, streaming responses, checkpoint selection, history, and professional guardrails.
- Quantization and export helpers.

## Installation

```bash
git clone https://github.com/<your-user>/mini_llm.git
cd mini_llm
python -m pip install -r requirements.txt
```

## Dataset

Build the local training dataset:

```bash
python data/raw/build_dataset.py --force
```

Expected outputs:

```text
data/raw/wikipedia.json
data/raw/gutenberg.json
data/raw/openassistant.json
data/raw/squad.json
data/raw/natural_dialogs.json
data/raw/instructions.json
data/raw/natural_responses.json
data/processed/dataset.jsonl
data/processed/train.txt
data/processed/val.txt
```

The generated dataset includes dialogue, QA, natural text, instruction examples, natural responses, and technical text. `data/raw/dataset_large.txt` is also generated for compatibility with the existing training pipeline.

## Tokenizer

```bash
python tokenizer/build_tokenizer.py \
  --dataset data/processed/train.txt \
  --out tokenizer/tokenizer.json \
  --vocab_size 8192
```

The tokenizer is byte-level, so it supports Italian characters, punctuation, code snippets, and arbitrary UTF-8 text.

## Training

Laptop-safe MiniLLM-32M training profile:

```bash
python run_training.py \
  --model_size mini_llm_32m \
  --seq_len 256 \
  --batch_size 1 \
  --epochs 8 \
  --gradient_checkpointing \
  --fp16 \
  --gpu_memory_fraction 0.70 \
  --gpu_max_temp 80 \
  --thermal_cooldown_seconds 10 \
  --eval_every 200 \
  --eval_batches 5 \
  --checkpoint_dir models/checkpoints
```

Training saves:

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

Open:

```text
http://127.0.0.1:8010
```

The dashboard reads logs, stats, checkpoint files, GPU metrics, benchmark results, and Chat Mode status.

## Chat Mode

```bash
python chat/server.py --checkpoint models/checkpoints/best.pt --port 8020
```

Open:

```text
http://127.0.0.1:8020/chat
```

API:

```http
POST /api/chat
GET /api/chat/checkpoints
```

Example:

```bash
curl -X POST http://127.0.0.1:8020/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Ciao, come ti chiami?\",\"temperature\":0.45,\"top_p\":0.82,\"max_tokens\":80,\"history\":[]}"
```

Chat Mode uses a local professional guardrail layer for simple questions, explanations, current-info limitations, writing tasks, and quality filtering before showing generated checkpoint output.

## Benchmark Suite

```bash
python evaluate.py --checkpoint models/checkpoints/best.pt
```

Benchmark results are saved in:

```text
data/benchmarks/results_<checkpoint>.json
```

Dashboard endpoint:

```http
POST /api/evaluate
Content-Type: application/json

{"checkpoint_path":"models/checkpoints/best.pt"}
```

Metrics:

- Perplexity: lower is better.
- Average log-likelihood: higher is better.
- Token accuracy: next-token exact match when expected answers are available.
- Coherence score: local heuristic for response readability.
- Repetition score: higher means less repetitive output.

## Project Structure

```text
model/          transformer and configs
tokenizer/      byte-level BPE tokenizer
training/       dataloaders, trainer, stats, controls
data/           raw and processed dataset tooling
chat/           local Chat Mode server and UI
benchmark/      benchmark dataset and evaluator
ui/             generation UI assets
utils/          quantization, plotting, helpers
tests/          unit and integration tests
export/         export artifacts
agents/         optional agent modules
plugins/        optional plugin modules
```

## Release

Current release target: `v1.0.0`.

```bash
git tag v1.0.0
```

## Roadmap

- Larger instruction-tuning dataset.
- LoRA adapters for cheaper fine-tuning.
- Streaming dataset training for corpora larger than RAM.
- More benchmark categories and regression gates.
- ONNX or TorchScript export.
- Model card and dataset card.

## License

MIT License. See [LICENSE](LICENSE).
