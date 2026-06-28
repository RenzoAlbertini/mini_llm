# Changelog

## [1.0.0] - 2026-06-28

### Added

- MiniLLM-32M transformer preset for local experiments.
- Professional dataset builder with Wikipedia, Gutenberg, OpenAssistant-style dialogue, SQuAD-style QA, instruction tuning, natural responses, and technical text.
- Byte-level BPE tokenizer export with `tokenizer.json`, `vocab.json`, and `merges.txt`.
- Safe laptop training profile with FP16, gradient checkpointing, checkpoint resume, and temperature-only cooldown.
- Local training dashboard with loss, GPU metrics, VRAM, logs, checkpoints, benchmark plots, and training controls.
- Benchmark Suite for checkpoint evaluation with perplexity, log-likelihood, token accuracy, coherence, and repetition metrics.
- Chat Mode with FastAPI, streaming responses, checkpoint selection, history, and a professional local guardrail layer.
- 4-bit quantization helpers and generation UI.

### Changed

- Training now prefers the generated dataset in `data/raw/dataset_large.txt` and processed outputs in `data/processed/`.
- GPU utilization-based throttling was removed; only temperature cooldown remains.
- Packaging now includes `chat` and `benchmark` modules.

### Notes

- The project is designed to run locally on consumer hardware.
- Large generated artifacts and checkpoints are excluded from Git by default.
