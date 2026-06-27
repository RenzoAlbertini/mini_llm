import argparse
import subprocess
import sys


def run_script(script, args):
    command = [sys.executable, script] + args
    return subprocess.call(command)


def main():
    parser = argparse.ArgumentParser(
        description="CLI completa per mini_llm.",
        epilog="Esempio: python cli.py train -- --demo",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    commands = {
        "train": ("run_training.py", "Training: python cli.py train -- --demo"),
        "finetune": ("finetune.py", "Fine-tuning: python cli.py finetune -- --base_checkpoint checkpoints/final.pt"),
        "generate": ("run_generate.py", "Generazione: python cli.py generate -- --prompt \"python is\""),
        "evaluate": ("evaluate_model.py", "Valutazione: python cli.py evaluate -- --checkpoint checkpoints/final.pt"),
        "benchmark": ("benchmark_inference.py", "Benchmark: python cli.py benchmark -- --checkpoint checkpoints/final.pt"),
        "export": ("export_model.py", "Export: python cli.py export -- --checkpoint checkpoints/final.pt"),
        "profile": ("profile_gpu.py", "Profiling: python cli.py profile -- --checkpoint checkpoints/final.pt"),
        "ui": ("ui_server.py", "UI web: python cli.py ui -- --port 8000"),
    }
    for name, (script, example) in commands.items():
        p = sub.add_parser(name, help=f"Esegue {script}", description=example)
        p.add_argument("args", nargs=argparse.REMAINDER, help="Argomenti pass-through per lo script. Usa -- prima degli argomenti.")

    args = parser.parse_args()
    passthrough = args.args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return run_script(commands[args.command][0], passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
