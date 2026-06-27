from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def _payload(self, model, optimizer=None, scheduler=None, config=None, step=0, epoch=0, val_loss=None):
        payload = {
            "model": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if config is not None:
            payload["config"] = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        return payload

    def save(self, name, model, optimizer=None, scheduler=None, config=None, step=0, epoch=0, val_loss=None):
        path = self.checkpoint_dir / name
        torch.save(self._payload(model, optimizer, scheduler, config, step, epoch, val_loss), path)
        return path

    def save_last(self, model, optimizer=None, scheduler=None, config=None, step=0, epoch=0, val_loss=None):
        return self.save("last.pt", model, optimizer, scheduler, config, step, epoch, val_loss)

    def save_best(self, model, optimizer=None, scheduler=None, config=None, step=0, epoch=0, val_loss=None):
        if val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = float("inf") if val_loss is None else val_loss
            return self.save("best.pt", model, optimizer, scheduler, config, step, epoch, val_loss)
        return None

    def save_step(self, model, optimizer=None, scheduler=None, config=None, step=0, epoch=0, val_loss=None):
        return self.save(f"step_{step}.pt", model, optimizer, scheduler, config, step, epoch, val_loss)

    def load(self, path, model=None, optimizer=None, scheduler=None, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        if model is not None:
            model.load_state_dict(checkpoint["model"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("val_loss") is not None:
            self.best_val_loss = min(self.best_val_loss, checkpoint["val_loss"])
        return checkpoint
