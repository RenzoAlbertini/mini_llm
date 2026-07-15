# Changelog

## [1.1.0] - 2026-07-15

### Fixed

- Replaced the broken, long-running Quick Start with an isolated 20-step smoke
  training that does not overwrite tracked dataset or tokenizer files.
- Fixed train/validation splitting for small datasets with anti-leakage gaps.
- Made generation greedy and deterministic by default, with opt-in seeded
  sampling and completion-only output.
- Corrected the sign handling of the repetition penalty and stopped penalizing
  prompt tokens as if they had already been generated.
- Made Chat Mode execute the selected checkpoint for every request and expose
  whether the final response came from the model or a post-model quality
  fallback.
- Removed silent demo fallback when a real UI checkpoint is missing.
- Made missing and incompatible checkpoints return clear errors.
- Fixed the test runner so dependency-skipped tests can no longer produce a
  false `OK` result.
- Fixed packaging so the installed `mini-llm` console command can find every
  root module and launch Chat Mode or the end-to-end pipeline.

### Changed

- Normalized legacy dataset labels to explicit synthetic provenance and added
  exact-text deduplication.
- Removed roughly 27 MB of generated synthetic JSON from the tracked tree; the
  deterministic builder now creates those ignored fixtures on demand.
- Unified conventional checkpoint paths under `models/checkpoints/`.
- Made every required pipeline step fail the pipeline when it fails.
- Updated the README with verified smoke, generation, chat, dataset provenance,
  and limitation instructions.

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
