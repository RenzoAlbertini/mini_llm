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

    def _load_model_state(self, checkpoint, model):
        checkpoint_state = checkpoint["model"]
        model_state = model.state_dict()
        compatible_state = {}
        adjusted = False
        skipped = []

        for name, value in checkpoint_state.items():
            if name not in model_state:
                continue
            target = model_state[name]
            if value.shape == target.shape:
                compatible_state[name] = value
                continue
            if (
                name == "position_embedding.weight"
                and value.ndim == 2
                and target.ndim == 2
                and value.shape[1] == target.shape[1]
                and value.shape[0] >= target.shape[0]
            ):
                resized = target.clone()
                resized.copy_(value[: target.shape[0]])
                compatible_state[name] = resized
                adjusted = True
                print(
                    "checkpoint adattato: position_embedding.weight "
                    f"{tuple(value.shape)} -> {tuple(target.shape)}"
                )
                continue
            skipped.append((name, tuple(value.shape), tuple(target.shape)))
            adjusted = True

        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        if skipped:
            for name, source_shape, target_shape in skipped:
                print(f"checkpoint: parametro saltato {name} {source_shape} -> {target_shape}")
        if missing:
            print(f"checkpoint: parametri inizializzati dal modello corrente: {len(missing)}")
            adjusted = True
        if unexpected:
            print(f"checkpoint: parametri inattesi ignorati: {len(unexpected)}")
            adjusted = True
        checkpoint["_model_state_adjusted"] = adjusted

    def load(self, path, model=None, optimizer=None, scheduler=None, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        if model is not None:
            self._load_model_state(checkpoint, model)
        adjusted = checkpoint.get("_model_state_adjusted", False)
        if optimizer is not None and "optimizer" in checkpoint and not adjusted:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif optimizer is not None and "optimizer" in checkpoint and adjusted:
            print("optimizer non ripreso: checkpoint adattato a una nuova seq_len")
        if scheduler is not None and "scheduler" in checkpoint and not adjusted:
            scheduler.load_state_dict(checkpoint["scheduler"])
        elif scheduler is not None and "scheduler" in checkpoint and adjusted:
            print("scheduler non ripreso: checkpoint adattato a una nuova seq_len")
        if checkpoint.get("val_loss") is not None:
            self.best_val_loss = min(self.best_val_loss, checkpoint["val_loss"])
        return checkpoint
