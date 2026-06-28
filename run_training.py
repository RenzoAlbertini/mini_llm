import argparse
import shutil
import sys
import time
from pathlib import Path

from config_manager import add_config_args, build_configs
from tokenizer.build_tokenizer import train_byte_bpe
from utils.plot_training import make_plots


class TeeLogger:
    def __init__(self, path, stream):
        self.stream = stream
        self.file = Path(path).open("a", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        if "\n" in text:
            self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Avvia il training reale di mini_llm.")
    add_config_args(parser)
    parser.add_argument("--demo", action="store_true", help="Run breve end-to-end con dataset demo.")
    parser.add_argument("--mode", choices=["debug", "standard", "production"], default="standard")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--data_source", action="append", default=[], help="Sorgente testo opzionale path:weight.")
    parser.add_argument("--max_train_chars", type=int, default=0, help="Limita i caratteri grezzi da tokenizzare. 0 = tutto.")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--processed", default="data/processed/real_tokens.pt")
    parser.add_argument("--checkpoint_dir", default="models/checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = nessun limite, usa tutte le epoche.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Forza mixed precision FP16 quando CUDA e disponibile.")
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        default=0.70,
        help="Frazione massima di VRAM usabile dal processo CUDA. 0 disabilita.",
    )
    parser.add_argument(
        "--gpu_duty_cycle",
        type=float,
        default=0.70,
        help="Compat legacy: ignorato. Il training usa solo controllo temperatura.",
    )
    parser.add_argument("--gpu_max_utilization", type=int, default=0, help="Compat legacy: ignorato.")
    parser.add_argument("--gpu_max_temp", type=int, default=80, help="Temperatura massima GPU prima di attendere. 0 disabilita.")
    parser.add_argument("--thermal_check_every", type=int, default=1, help="Controlla termiche ogni N step.")
    parser.add_argument("--thermal_cooldown_seconds", type=float, default=10.0, help="Pausa quando GPU supera soglie termiche.")
    parser.add_argument("--no_dynamic_batch", action="store_true", help="Disabilita riduzione automatica batch su CUDA.")
    parser.add_argument("--no_dynamic_context", action="store_true", help="Disabilita riduzione automatica seq_len su CUDA.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no_resume", action="store_true", help="Disabilita il resume automatico da last checkpoint.")
    parser.add_argument("--stats_path", default="data/logs/training_stats.csv")
    parser.add_argument("--log_path", default="data/logs/training.log")
    parser.add_argument("--plots_dir", default="data/plots")
    parser.add_argument("--profile", action="store_true", help="Salva un report torch.profiler in data/profiling/.")
    parser.add_argument("--save_quantized", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def make_config(args):
    config, _ = build_configs(args)
    config.use_gradient_checkpointing = args.gradient_checkpointing
    return config


def apply_mode(args):
    if args.mode == "debug":
        args.batch_size = min(args.batch_size, 2)
        args.max_steps = min(args.max_steps, 20)
        args.eval_every = min(args.eval_every, 10)
        args.log_every = 1
        args.seq_len = min(args.seq_len or 64, 64)
        print("modalita debug: batch piccolo, seq_len corto, logging dettagliato")
    elif args.mode == "production":
        args.log_every = max(args.log_every, 50)
        args.eval_every = max(args.eval_every, 200)
        print("modalita production: logging ridotto e checkpoint ordinati")


def prepare_demo_assets(args):
    from data.raw.prepare_dataset import SAMPLE_TEXT, clean_text

    print("modalita demo: preparo dataset piccolo e run breve di prova")
    args.data_dir = "data/raw"
    args.processed = "data/processed/demo_tokens.pt"
    args.checkpoint_dir = "checkpoints/demo"
    args.stats_path = "data/logs/demo_training_stats.csv"
    args.plots_dir = "data/plots/demo"
    args.batch_size = min(args.batch_size, 4)
    args.epochs = 1
    args.max_steps = min(args.max_steps, 20)
    args.eval_every = min(args.eval_every, 10)
    args.log_every = 1

    dataset_path = Path(args.data_dir) / "dataset.txt"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(clean_text(SAMPLE_TEXT), encoding="utf-8")

    tokenizer_path = Path(args.tokenizer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = train_byte_bpe(dataset_path.read_text(encoding="utf-8"), vocab_size=512)
    tokenizer.save_model(tokenizer_path)

    processed_path = Path(args.processed)
    if processed_path.exists():
        processed_path.unlink()


def ensure_real_assets(args):
    data_dir = Path(args.data_dir)
    large_dataset = data_dir / "dataset_large.txt"
    default_dataset = data_dir / "dataset.txt"
    dataset_path = large_dataset if large_dataset.exists() else default_dataset
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {dataset_path}. "
            "Crea il dataset con: python data/raw/prepare_dataset.py --out data/raw/dataset.txt "
            "oppure python data/raw/build_dataset.py"
        )
    if large_dataset.exists():
        args.data_dir = str(large_dataset)
        if Path(args.processed).name == "real_tokens.pt":
            args.processed = "data/processed/large_tokens.pt"
        print(f"dataset grande rilevato: {large_dataset}")

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print("tokenizer non trovato: lo costruisco dal dataset reale")
        text = dataset_path.read_text(encoding="utf-8", errors="ignore")
        tokenizer = train_byte_bpe(text, vocab_size=args.vocab_size or 8192)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(tokenizer_path)


def apply_training_defaults(args):
    if args.scheduler is None:
        args.scheduler = "cosine"
    if args.warmup_steps is None:
        args.warmup_steps = 100
    if args.min_lr is None:
        args.min_lr = args.lr * 0.1

    is_32m = getattr(args, "model_size", None) == "mini_llm_32m"
    last_name = "mini_llm_32m_last.pt" if is_32m else "last.pt"
    last_checkpoint = Path(args.checkpoint_dir) / last_name
    if args.no_resume:
        args.resume = None
        return
    if args.resume is None and last_checkpoint.exists():
        tokenizer_path = Path(args.tokenizer)
        if tokenizer_path.exists() and tokenizer_path.stat().st_mtime > last_checkpoint.stat().st_mtime:
            print(
                "resume automatico saltato: tokenizer piu recente del checkpoint. "
                "Avvio training da zero per evitare mismatch tokenizer/checkpoint."
            )
            return
        args.resume = str(last_checkpoint)
        print(f"resume automatico da {args.resume}")


def publish_named_checkpoints(args):
    if getattr(args, "model_size", None) != "mini_llm_32m":
        return []
    checkpoint_dir = Path(args.checkpoint_dir)
    mapping = {
        "best.pt": "mini_llm_32m_best.pt",
        "last.pt": "mini_llm_32m_last.pt",
    }
    published = []
    for src_name, dst_name in mapping.items():
        src = checkpoint_dir / src_name
        dst = checkpoint_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            published.append(dst)
    return published


def main():
    args = build_parser().parse_args()
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(args.log_path, sys.stdout)
    original_stdout = sys.stdout
    sys.stdout = logger
    started = time.perf_counter()
    try:
        _main(args)
    finally:
        elapsed = time.perf_counter() - started
        print(f"log salvato in {args.log_path}")
        print(f"tempo totale run_training: {elapsed:.2f}s")
        sys.stdout = original_stdout
        logger.close()


def _main(args):
    if args.demo:
        prepare_demo_assets(args)
    else:
        ensure_real_assets(args)
    apply_training_defaults(args)
    apply_mode(args)

    config = make_config(args)
    mode = "demo" if args.demo else args.mode
    print(f"modalita: {mode}")
    print(f"config: {config}")
    print(
        f"training: batch_size={args.batch_size} | epochs={args.epochs} | "
        f"max_steps={args.max_steps} | lr={args.lr} | scheduler={args.scheduler} | "
        f"warmup_steps={args.warmup_steps} | patience={args.patience} | "
        f"gpu_memory_fraction={args.gpu_memory_fraction} | gpu_max_temp={args.gpu_max_temp}"
    )

    try:
        from training.train import train_model
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("PyTorch non installato. Installa le dipendenze con: pip install -r requirements.txt")
            return
        raise

    final_path = train_model(args, config=config)
    for path in publish_named_checkpoints(args):
        print(f"checkpoint 32M pubblicato: {path}")
    make_plots(args.stats_path, args.plots_dir)
    print(f"grafici salvati in {args.plots_dir}")

    if args.profile:
        run_profile(args, final_path)

    if args.save_quantized:
        from model.config import ModelConfig
        from model.transformer import MiniTransformerLM
        from utils.helpers import get_device, load_checkpoint
        from utils.quantization import save_quantized_model

        device = get_device()
        checkpoint = load_checkpoint(final_path, device=device)
        model = MiniTransformerLM(ModelConfig.from_dict(checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model"])
        quantized_path = f"{args.checkpoint_dir}/final_quantized.pt"
        save_quantized_model(model, quantized_path, config=model.config)
        print(f"checkpoint quantizzato salvato: {quantized_path}")

    print("training completato")


def run_profile(args, checkpoint_path):
    try:
        import torch

        from model.config import ModelConfig
        from model.transformer import MiniTransformerLM
        from tokenizer.tokenizer import BPETokenizer
        from training.dataset import create_dataloaders, load_or_tokenize
        from utils.helpers import get_device, gpu_profile, load_checkpoint
    except ModuleNotFoundError as exc:
        print(f"profiling saltato: dipendenza mancante {exc.name}")
        return

    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model = MiniTransformerLM(ModelConfig.from_dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    tokens = load_or_tokenize(args.data_dir, args.tokenizer, args.processed, sources=args.data_source)
    train_loader, _ = create_dataloaders(tokens, model.config.seq_len, args.batch_size, val_fraction=args.val_fraction)
    sample = next(iter(train_loader))
    report = gpu_profile(model, sample, out_dir="data/profiling", steps=3)
    print(f"profiling salvato in {report}")


if __name__ == "__main__":
    main()
