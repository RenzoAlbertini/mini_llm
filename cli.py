import argparse
import subprocess
import sys


def run_module(module, args):
    command = [sys.executable, "-m", module] + args
    return subprocess.call(command)


def main():
    parser = argparse.ArgumentParser(
        description="CLI completa per mini_llm.",
        epilog="Esempio: mini-llm train -- --demo",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    commands = {
        "train": ("run_training", "Training: mini-llm train -- --demo"),
        "finetune": ("finetune", "Fine-tuning: mini-llm finetune -- --base_checkpoint models/checkpoints/best.pt"),
        "generate": ("run_generate", "Generazione: mini-llm generate -- --prompt \"python is\""),
        "evaluate": ("evaluate_model", "Valutazione: mini-llm evaluate -- --checkpoint models/checkpoints/best.pt"),
        "benchmark": ("benchmark_inference", "Benchmark: mini-llm benchmark -- --checkpoint models/checkpoints/best.pt"),
        "export": ("export_model", "Export: mini-llm export -- --checkpoint models/checkpoints/best.pt"),
        "profile": ("profile_gpu", "Profiling: mini-llm profile -- --checkpoint models/checkpoints/best.pt"),
        "chat": ("chat.server", "Chat: mini-llm chat -- --checkpoint models/checkpoints/best.pt"),
        "ui": ("ui_server", "UI web: mini-llm ui -- --port 8000"),
        "pipeline": ("pipeline", "Pipeline smoke: mini-llm pipeline"),
    }
    for name, (script, example) in commands.items():
        p = sub.add_parser(name, help=f"Esegue {script}", description=example)
        p.add_argument("args", nargs=argparse.REMAINDER, help="Argomenti pass-through per lo script. Usa -- prima degli argomenti.")

    args = parser.parse_args()
    passthrough = args.args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return run_module(commands[args.command][0], passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
