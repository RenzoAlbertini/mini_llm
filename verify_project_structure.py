import importlib
from pathlib import Path


REQUIRED_FILES = [
    "tokenizer/build_tokenizer.py",
    "tokenizer/tokenizer.py",
    "model/config.py",
    "model/transformer.py",
    "training/dataset.py",
    "training/train.py",
    "training/training_stats.py",
    "inference/generate.py",
    "utils/helpers.py",
    "utils/quantization.py",
    "utils/plot_training.py",
    "config_manager.py",
    "checkpoint_manager.py",
    "data/raw/prepare_dataset.py",
    "tests/test_tokenizer.py",
    "tests/test_model_forward.py",
    "tests/test_generate.py",
    "tests/test_end_to_end.py",
    "run_training.py",
    "run_generate.py",
    "sanity_check.py",
    "run_all_tests.py",
    "pre_training_check.py",
    "evaluate_model.py",
    "benchmark_inference.py",
    "export_model.py",
    "finetune.py",
    "pipeline.py",
    "api_server.py",
    "cli.py",
    "demo_interactive.py",
    "validate_model.py",
    "stress_test.py",
    "profile_gpu.py",
    "ui_server.py",
    "ui/index.html",
    "ui/style.css",
    "ui/app.js",
    "setup.py",
    "pyproject.toml",
    "mini_llm.py",
    "README.md",
]

MODULES = [
    "tokenizer.tokenizer",
    "tokenizer.build_tokenizer",
    "model.config",
    "model.transformer",
    "training.dataset",
    "training.train",
    "training.training_stats",
    "inference.generate",
    "utils.helpers",
    "utils.quantization",
    "utils.plot_training",
    "config_manager",
    "checkpoint_manager",
    "run_training",
    "run_generate",
    "run_all_tests",
    "sanity_check",
    "pre_training_check",
    "evaluate_model",
    "benchmark_inference",
    "export_model",
    "finetune",
    "pipeline",
    "api_server",
    "cli",
    "demo_interactive",
    "validate_model",
    "stress_test",
    "profile_gpu",
    "ui_server",
    "mini_llm",
]


def check_file(path):
    return Path(path).exists()


def check_import(module_name):
    try:
        importlib.import_module(module_name)
        return "OK", ""
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return "MISSING_DEP", "PyTorch non installato"
        return "FAIL", f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return "FAIL", f"{type(exc).__name__}: {exc}"


def main():
    print("mini_llm project verification")
    print("=============================")

    ok = True
    print("\nFiles")
    for path in REQUIRED_FILES:
        exists = check_file(path)
        ok = ok and exists
        status = "OK" if exists else "FAIL"
        print(f"{status:4s} {path}")

    print("\nImports")
    for module_name in MODULES:
        status, error = check_import(module_name)
        ok = ok and status in {"OK", "MISSING_DEP"}
        suffix = "" if status == "OK" else f" -> {error}"
        print(f"{status:4s} {module_name}{suffix}")

    print("\nSummary")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
