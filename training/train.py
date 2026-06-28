import argparse
import atexit
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast

from checkpoint_manager import CheckpointManager
from config_manager import TrainingConfig, save_run_config
from memory_manager import MemoryManager
from model.config import ModelConfig
from model.transformer import MiniTransformerLM
from tokenizer.tokenizer import BPETokenizer
from training.dataset import create_dataloaders, load_or_tokenize
from training.training_stats import TrainingStats
from training.control import clear_pid, wait_while_paused, write_control, write_pid
from utils.helpers import (
    count_parameters,
    get_amp_dtype,
    get_device,
    memory_usage,
    set_seed,
    timer,
)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20, amp_dtype=None):
    model.eval()
    losses = []
    use_amp = device.type == "cuda" and amp_dtype is not None
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with autocast(enabled=True, dtype=amp_dtype) if use_amp else nullcontext():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Training minimalista di un mini Transformer LM.")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--processed", default="data/processed/tokens.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=0, help="Checkpoint extra ogni N step. 0 disabilita.")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping dopo N validazioni senza miglioramento.")
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
    parser.add_argument("--stats_path", default="data/logs/training_stats.csv")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum_start_seq_len", type=int, default=64)
    parser.add_argument("--curriculum_step_size", type=int, default=200)
    parser.add_argument("--curriculum_increment", type=int, default=64)
    parser.add_argument("--data_source", action="append", default=[], help="Sorgente testo opzionale path:weight.")
    parser.add_argument("--max_train_chars", type=int, default=0, help="Limita i caratteri grezzi da tokenizzare. 0 = tutto.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def get_arg(args, name, default):
    return getattr(args, name, default)


def build_scheduler(optimizer, args):
    scheduler_name = get_arg(args, "scheduler", "cosine")
    if scheduler_name == "none":
        return None

    max_steps = max(1, get_arg(args, "max_steps", 1000))
    warmup_steps = max(0, get_arg(args, "warmup_steps", 100))
    min_lr = get_arg(args, "min_lr", 3e-5)
    base_lr = optimizer.param_groups[0]["lr"]
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_name == "linear":
            return max(min_factor, 1.0 - progress * (1.0 - min_factor))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_curriculum_seq_len(args, step, max_seq_len):
    if not get_arg(args, "curriculum", False):
        return max_seq_len
    start = min(max_seq_len, max(1, get_arg(args, "curriculum_start_seq_len", 64)))
    interval = max(1, get_arg(args, "curriculum_step_size", 200))
    increment = max(1, get_arg(args, "curriculum_increment", 64))
    seq_len = start + (step // interval) * increment
    return min(max_seq_len, seq_len)


def read_nvidia_smi_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = result.stdout.strip().splitlines()[0]
    values = [value.strip() for value in line.split(",")]
    if not values:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def thermal_throttle(args, device, step):
    if device.type != "cuda":
        return
    check_every = max(1, int(get_arg(args, "thermal_check_every", 1)))
    if step % check_every != 0:
        return
    max_temp = int(get_arg(args, "gpu_max_temp", 80) or 0)
    if max_temp <= 0:
        return
    cooldown = max(1.0, float(get_arg(args, "thermal_cooldown_seconds", 10.0)))
    while True:
        temp = read_nvidia_smi_status()
        if temp is None:
            return
        if temp <= max_temp:
            return
        print(f"raffreddamento GPU: temp={temp}C (target <= {max_temp}C), pausa {cooldown:.1f}s")
        time.sleep(cooldown)


def train_model(args, config=None):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)

    set_seed(args.seed)
    write_control(paused=False, stop_requested=False)
    write_pid()
    atexit.register(clear_pid)
    device = get_device()
    gpu_memory_fraction = float(get_arg(args, "gpu_memory_fraction", 0.70) or 0.0)
    if device.type == "cuda" and gpu_memory_fraction > 0.0:
        gpu_memory_fraction = min(max(gpu_memory_fraction, 0.05), 1.0)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=device_index)
        print(f"limite VRAM processo CUDA: {gpu_memory_fraction:.0%}")
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    config = config or ModelConfig(vocab_size=tokenizer.vocab_size)
    config.vocab_size = tokenizer.vocab_size
    config.use_gradient_checkpointing = bool(get_arg(args, "gradient_checkpointing", False))
    memory = MemoryManager(device)
    if not get_arg(args, "no_dynamic_context", False):
        suggested_seq_len = memory.recommend_seq_len(config.seq_len, config.d_model, config.n_layers, args.batch_size)
        if suggested_seq_len < config.seq_len:
            print(f"context dinamico: seq_len {config.seq_len} -> {suggested_seq_len}")
            config.seq_len = suggested_seq_len
    if not get_arg(args, "no_dynamic_batch", False):
        suggested_batch = memory.recommend_batch_size(args.batch_size, config.seq_len, config.d_model, config.n_layers)
        if suggested_batch < args.batch_size:
            print(f"batch dinamico: batch_size {args.batch_size} -> {suggested_batch}")
            args.batch_size = suggested_batch
    training_config = TrainingConfig(
        lr=args.lr,
        min_lr=get_arg(args, "min_lr", 3e-5),
        warmup_steps=get_arg(args, "warmup_steps", 100),
        scheduler=get_arg(args, "scheduler", "cosine"),
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        patience=args.patience,
        curriculum=get_arg(args, "curriculum", False),
        curriculum_start_seq_len=get_arg(args, "curriculum_start_seq_len", 64),
        curriculum_step_size=get_arg(args, "curriculum_step_size", 200),
        curriculum_increment=get_arg(args, "curriculum_increment", 64),
        gpu_memory_fraction=gpu_memory_fraction,
        gpu_max_temp=get_arg(args, "gpu_max_temp", 80),
        thermal_check_every=get_arg(args, "thermal_check_every", 1),
        thermal_cooldown_seconds=get_arg(args, "thermal_cooldown_seconds", 10.0),
    )
    save_run_config(Path(args.checkpoint_dir) / "config.json", config, training_config)

    tokens = load_or_tokenize(
        args.data_dir,
        args.tokenizer,
        args.processed,
        sources=get_arg(args, "data_source", []),
        max_chars=get_arg(args, "max_train_chars", 0),
    )
    train_loader, val_loader = create_dataloaders(
        tokens,
        config.seq_len,
        args.batch_size,
        val_fraction=args.val_fraction,
        stride=args.stride,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = MiniTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    precision = memory.choose_precision()
    if get_arg(args, "fp16", False) and device.type == "cuda":
        precision.use_amp = True
        precision.dtype = torch.float16
        precision.name = "fp16"
    use_amp = precision.use_amp
    amp_dtype = precision.dtype
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    stats = TrainingStats(args.stats_path)
    checkpoints = CheckpointManager(args.checkpoint_dir)

    print(f"device={device} | parametri={count_parameters(model) / 1e6:.2f}M")
    if amp_dtype is not None:
        print(f"mixed precision: {precision.name}")
    print(f"memoria iniziale: {memory_usage(device)}")
    model.train()
    step = 0
    start_epoch = 1
    best_val_loss = float("inf")
    patience_count = 0

    if get_arg(args, "resume", None):
        loaded = checkpoints.load(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, device=device)
        step = int(loaded.get("step", 0))
        start_epoch = int(loaded.get("epoch", 0)) + 1
        best_val_loss = checkpoints.best_val_loss
        print(f"resume da {args.resume} | step={step} | start_epoch={start_epoch}")

    with timer("training"):
        last_seq_len = None
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.perf_counter()
            running_loss = 0.0
            batches = 0
            for x, y in train_loader:
                batch_start = time.perf_counter()
                step += 1
                batches += 1
                wait_while_paused()
                thermal_throttle(args, device, step)
                x = x.to(device)
                y = y.to(device)
                seq_len = current_curriculum_seq_len(args, step, config.seq_len)
                if seq_len != last_seq_len:
                    print(f"curriculum seq_len={seq_len}/{config.seq_len} a step {step}")
                    last_seq_len = seq_len
                if seq_len < x.shape[1]:
                    x = x[:, :seq_len]
                    y = y[:, :seq_len]

                micro_batch_size = x.size(0)
                while True:
                    try:
                        optimizer.zero_grad(set_to_none=True)
                        total_loss = 0.0
                        chunks = 0
                        for start in range(0, x.size(0), micro_batch_size):
                            xb = x[start:start + micro_batch_size]
                            yb = y[start:start + micro_batch_size]
                            chunks += 1
                            with autocast(enabled=True, dtype=amp_dtype) if use_amp else nullcontext():
                                _, chunk_loss = model(xb, yb)
                                scaled_loss = chunk_loss / max(1, (x.size(0) + micro_batch_size - 1) // micro_batch_size)
                            scaler.scale(scaled_loss).backward()
                            total_loss += chunk_loss.detach().item() * (xb.size(0) / x.size(0))
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        loss_value = total_loss
                        break
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower() or device.type != "cuda":
                            raise
                        memory.empty_cache()
                        if micro_batch_size > 1:
                            micro_batch_size = max(1, micro_batch_size // 2)
                            print(f"OOM CUDA: retry micro_batch_size={micro_batch_size}")
                            continue
                        fallback = memory.fallback_precision(amp_dtype)
                        if fallback.dtype != amp_dtype or fallback.use_amp != use_amp:
                            precision = fallback
                            use_amp = precision.use_amp
                            amp_dtype = precision.dtype
                            scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
                            print(f"OOM CUDA: fallback precision={precision.name}")
                            continue
                        raise
                if scheduler is not None:
                    scheduler.step()

                running_loss += loss_value
                lr = optimizer.param_groups[0]["lr"]
                stats.log(step, epoch, train_loss=loss_value, lr=lr)
                batch_time = time.perf_counter() - batch_start
                thermal_throttle(args, device, step)

                if step % args.log_every == 0:
                    avg_loss = running_loss / max(1, batches)
                    print(
                        f"epoch {epoch:02d} | step {step:05d} | "
                        f"loss={loss_value:.4f} | avg_loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | batch_time={batch_time:.3f}s"
                    )

                if step % args.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device, max_batches=get_arg(args, "eval_batches", 20), amp_dtype=amp_dtype)
                    stats.log(step, epoch, val_loss=val_loss, lr=lr)
                    print(f"step {step:05d} | val_loss={val_loss:.4f} | memoria={memory_usage(device)}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_count = 0
                        checkpoints.save_best(model, optimizer, scheduler, config, step, epoch, val_loss)
                    else:
                        patience_count += 1
                    if args.patience > 0 and patience_count >= args.patience:
                        print("early stopping attivato")
                        final_path = Path(args.checkpoint_dir) / "final.pt"
                        checkpoints.save("final.pt", model, optimizer, scheduler, config, step, epoch, val_loss)
                        return final_path

                if args.save_every > 0 and step % args.save_every == 0:
                    ckpt_path = checkpoints.save_step(model, optimizer, scheduler, config, step, epoch, val_loss=None)
                    print(f"checkpoint salvato: {ckpt_path}")

                if args.max_steps > 0 and step >= args.max_steps:
                    break

            avg_loss = running_loss / max(1, batches)
            val_loss = evaluate(model, val_loader, device, max_batches=get_arg(args, "eval_batches", 20), amp_dtype=amp_dtype)
            stats.log(step, epoch, train_loss=avg_loss, val_loss=val_loss, lr=optimizer.param_groups[0]["lr"])
            epoch_path = Path(args.checkpoint_dir) / f"epoch_{epoch}.pt"
            checkpoints.save(f"epoch_{epoch}.pt", model, optimizer, scheduler, config, step, epoch, val_loss)
            checkpoints.save_last(model, optimizer, scheduler, config, step, epoch, val_loss)
            epoch_time = time.perf_counter() - epoch_start
            print(
                f"epoch {epoch:02d} finita | avg_loss={avg_loss:.4f} | "
                f"val_loss={val_loss:.4f} | epoch_time={epoch_time:.2f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                checkpoints.save_best(model, optimizer, scheduler, config, step, epoch, val_loss)
            else:
                patience_count += 1
            if args.patience > 0 and patience_count >= args.patience:
                print("early stopping attivato")
                break
            if args.max_steps > 0 and step >= args.max_steps:
                break

    final_path = Path(args.checkpoint_dir) / "final.pt"
    val_loss = evaluate(model, val_loader, device, max_batches=get_arg(args, "eval_batches", 20), amp_dtype=amp_dtype)
    checkpoints.save("final.pt", model, optimizer, scheduler, config, step, args.epochs, val_loss)
    print(f"training finito | val_loss={val_loss:.4f} | checkpoint={final_path}")
    clear_pid()
    return final_path


def main():
    args = build_arg_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
