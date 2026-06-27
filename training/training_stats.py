import csv
import time
from pathlib import Path


class TrainingStats:
    def __init__(self, out_path="data/logs/training_stats.csv"):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self._last_time = time.perf_counter()

    def log(self, step, epoch, train_loss=None, val_loss=None, lr=None):
        now = time.perf_counter()
        row = {
            "step": step,
            "epoch": epoch,
            "train_loss": "" if train_loss is None else float(train_loss),
            "val_loss": "" if val_loss is None else float(val_loss),
            "lr": "" if lr is None else float(lr),
            "step_time": now - self._last_time,
        }
        self._last_time = now
        self.rows.append(row)
        self.save()

    def save(self):
        fieldnames = ["step", "epoch", "train_loss", "val_loss", "lr", "step_time"]
        with self.out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
