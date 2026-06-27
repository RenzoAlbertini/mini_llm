# Mini-LLM — Lightweight Local Language Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Mini-LLM is a compact, educational local language-model framework built with PyTorch. It includes a byte-level BPE tokenizer, a small decoder-only Transformer, full training and inference scripts, a web UI with token streaming, API endpoints, benchmarking, stress testing, quantization utilities, export tools, and a simple plugin/agent-ready project layout.

The goal is not to compete with production LLM stacks. The goal is to make every moving piece understandable, hackable, and runnable on a local machine.

## Screenshot

![Mini-LLM Web UI](/ui/screenshot.png)

## Features

- Full training loop with FP32, FP16, and BF16 support where available
- 8-bit linear weight quantization utilities
- Web UI with streaming token generation
- FastAPI server and WebSocket streaming endpoint
- Professional CLI wrapper
- End-to-end pipeline runner
- Plugin system-ready project structure
- Local agents-ready project structure
- Metrics dashboard for token/sec, RAM, and VRAM
- Stress test runner
- GPU profiling hooks
- Export to a minimal `safetensors` layout
- Fine-tuning on custom datasets
- Checkpoint manager with best/last/final checkpoints
- Warmup + cosine learning-rate scheduler
- Early stopping and resume support

## Project Structure

```text
mini_llm/
  model/              Transformer model and config
  tokenizer/          Byte-level BPE tokenizer
  training/           Dataset, training loop, stats
  inference/          Text generation
  utils/              Helpers, quantization, plotting
  ui/                 Web UI assets
  plugins/            Plugin extension point
  agents/             Local agent extension point
  data/               Raw data, logs, plots, reports
  tests/              Unit and end-to-end tests
  export/             Export output folder
```

## Installation

```bash
git clone https://github.com/<your-user>/mini_llm.git
cd mini_llm
python -m pip install -r requirements.txt
```

Editable install:

```bash
python -m pip install -e .
```

## Quickstart

Run the full local verification:

```bash
python verify_project_structure.py
python run_all_tests.py
python sanity_check.py
```

Run the end-to-end demo pipeline:

```bash
python pipeline.py
```

## Web UI

Start the web UI:

```bash
python ui_server.py --checkpoint models/checkpoints/final.pt --tokenizer tokenizer/tokenizer.json
```

Open:

```text
http://127.0.0.1:8000/ui
```

The UI supports:

- prompt input
- streaming token output over WebSocket
- temperature, top-k, top-p, and max token controls
- FP32 / FP16 / BF16 / 8-bit / demo model selector
- metrics panel
- agents panel
- plugins panel

## Training

Prepare a dataset:

```bash
python data/raw/prepare_dataset.py --out data/raw/dataset.txt
```

Build the tokenizer:

```bash
python -m tokenizer.build_tokenizer --data_dir data/raw --out tokenizer/tokenizer.json --vocab_size 512
```

Run real training:

```bash
python run_training.py --epochs 3 --batch_size 8 --lr 3e-4 --seq_len 128
```

Outputs:

```text
models/checkpoints/best.pt
models/checkpoints/last.pt
models/checkpoints/final.pt
data/logs/training.log
data/plots/train_loss.png
data/plots/val_loss.png
data/plots/learning_rate.png
```

## Inference

Generate from the trained model:

```bash
python run_generate.py --checkpoint models/checkpoints/final.pt --prompt "python is"
```

Stream output:

```bash
python run_generate.py --checkpoint models/checkpoints/final.pt --prompt "python is" --stream
```

Guided generation:

```bash
python run_generate.py \
  --checkpoint models/checkpoints/final.pt \
  --prompt "python is" \
  --required_word tensor \
  --repetition_penalty 1.2 \
  --num_samples 3
```

## API Server

Start the local API:

```bash
python api_server.py --checkpoint models/checkpoints/final.pt --tokenizer tokenizer/tokenizer.json
```

Endpoints:

- `GET /health`
- `GET /info`
- `POST /generate`
- `POST /evaluate`

## CLI

The CLI forwards commands to the dedicated scripts:

```bash
python cli.py train -- --epochs 3 --batch_size 8
python cli.py generate -- --checkpoint models/checkpoints/final.pt --prompt "python is"
python cli.py evaluate -- --checkpoint models/checkpoints/final.pt
python cli.py benchmark -- --checkpoint models/checkpoints/final.pt
python cli.py export -- --checkpoint models/checkpoints/final.pt
python cli.py ui -- --checkpoint models/checkpoints/final.pt
```

## Benchmark

```bash
python benchmark_inference.py --checkpoint models/checkpoints/final.pt --max_new_tokens 50
```

Benchmark reports are saved to:

```text
data/benchmarks/
```

## Export

Export model, tokenizer, config, manifest, and safetensors:

```bash
python export_model.py --checkpoint models/checkpoints/final.pt --tokenizer tokenizer/tokenizer.json --out_dir export
```

## Fine-Tuning

```bash
python finetune.py \
  --base_checkpoint models/checkpoints/final.pt \
  --data_dir data/raw \
  --checkpoint_dir models/checkpoints/finetune
```

## Stress Test

```bash
python stress_test.py --checkpoint models/checkpoints/final.pt --prompts 100
```

## GPU Profiling

```bash
python profile_gpu.py --checkpoint models/checkpoints/final.pt
```

Reports are written to:

```text
data/profiling/
```

## Plugin System

The `plugins/` folder is reserved for local extensions. The web UI can discover plugins placed there.

## Local Agents

The `agents/` folder is reserved for local agent definitions. The web UI can discover agents placed there.

## Roadmap

- Add richer tokenizer training options
- Add dataset streaming for larger corpora
- Add ONNX export
- Add richer plugin contracts
- Add agent configuration examples
- Add optional experiment tracking
- Add model cards for exported checkpoints

## License

MIT License. See [LICENSE](LICENSE).
