import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast

from checkpoint_manager import CheckpointManager
from config_manager import TrainingConfig, save_run_config
from model.config import ModelConfig
from model.transformer import MiniTransformerLM
from tokenizer.tokenizer import BPETokenizer
from training.dataset import create_dataloaders, load_or_tokenize
from training.training_stats import TrainingStats
from utils.helpers import (
    count_parameters,
    get_amp_dtype,
    get_device,
    memory_usage,
    set_seed,
    timer,
)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
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
    parser.add_argument("--save_every", type=int, default=0, help="Checkpoint extra ogni N step. 0 disabilita.")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping dopo N validazioni senza miglioramento.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
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


def train_model(args, config=None):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)

    set_seed(args.seed)
    device = get_device()
    tokenizer = BPETokenizer.load_model(args.tokenizer)
    config = config or ModelConfig(vocab_size=tokenizer.vocab_size)
    config.vocab_size = tokenizer.vocab_size
    config.use_gradient_checkpointing = bool(get_arg(args, "gradient_checkpointing", False))
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
    )
    save_run_config(Path(args.checkpoint_dir) / "config.json", config, training_config)

    tokens = load_or_tokenize(
        args.data_dir,
        args.tokenizer,
        args.processed,
        sources=get_arg(args, "data_source", []),
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
    use_amp = device.type == "cuda"
    amp_dtype = get_amp_dtype() if use_amp else None
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    stats = TrainingStats(args.stats_path)
    checkpoints = CheckpointManager(args.checkpoint_dir)

    print(f"device={device} | parametri={count_parameters(model) / 1e6:.2f}M")
    if amp_dtype is not None:
        print(f"mixed precision: {str(amp_dtype).replace('torch.', '')}")
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
                x = x.to(device)
                y = y.to(device)
                seq_len = current_curriculum_seq_len(args, step, config.seq_len)
                if seq_len != last_seq_len:
                    print(f"curriculum seq_len={seq_len}/{config.seq_len} a step {step}")
                    last_seq_len = seq_len
                if seq_len < x.shape[1]:
                    x = x[:, :seq_len]
                    y = y[:, :seq_len]

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=True, dtype=amp_dtype) if use_amp else nullcontext():
                    _, loss = model(x, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

                running_loss += loss.item()
                lr = optimizer.param_groups[0]["lr"]
                stats.log(step, epoch, train_loss=loss.item(), lr=lr)
                batch_time = time.perf_counter() - batch_start

                if step % args.log_every == 0:
                    avg_loss = running_loss / max(1, batches)
                    print(
                        f"epoch {epoch:02d} | step {step:05d} | "
                        f"loss={loss.item():.4f} | avg_loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | batch_time={batch_time:.3f}s"
                    )

                if step % args.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
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
            val_loss = evaluate(model, val_loader, device)
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
    val_loss = evaluate(model, val_loader, device)
    checkpoints.save("final.pt", model, optimizer, scheduler, config, step, args.epochs, val_loss)
    print(f"training finito | val_loss={val_loss:.4f} | checkpoint={final_path}")
    return final_path


def main():
    args = build_arg_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
