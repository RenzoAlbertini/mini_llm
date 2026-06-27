import argparse
import csv
import math
import struct
import zlib
from pathlib import Path


def read_stats(csv_path):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _save_png(path, width, height, pixels):
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        row = pixels[y]
        for r, g, b in row:
            raw.extend([r, g, b])
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    png.extend(_chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def _line(pixels, x0, y0, x1, y1, color):
    width = len(pixels[0])
    height = len(pixels)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def plot_series(points, out_path, width=900, height=420):
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
    margin = 40
    for x in range(margin, width - margin):
        pixels[height - margin][x] = (210, 210, 210)
    for y in range(margin, height - margin):
        pixels[y][margin] = (210, 210, 210)

    clean = [(x, y) for x, y in points if y is not None and math.isfinite(y)]
    if len(clean) >= 2:
        xs = [p[0] for p in clean]
        ys = [p[1] for p in clean]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if min_y == max_y:
            max_y = min_y + 1.0
        prev = None
        for x, y in clean:
            px = margin + int((x - min_x) / max(1, max_x - min_x) * (width - 2 * margin))
            py = height - margin - int((y - min_y) / (max_y - min_y) * (height - 2 * margin))
            if prev:
                _line(pixels, prev[0], prev[1], px, py, (30, 90, 180))
            prev = (px, py)
    _save_png(out_path, width, height, pixels)


def make_plots(csv_path="data/logs/training_stats.csv", out_dir="data/plots"):
    rows = read_stats(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = [(int(r["step"]), float(r["train_loss"])) for r in rows if r.get("train_loss")]
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r.get("val_loss")]
    times = [(int(r["step"]), float(r["step_time"])) for r in rows if r.get("step_time")]
    lrs = [(int(r["step"]), float(r["lr"])) for r in rows if r.get("lr")]
    if train:
        plot_series(train, out_dir / "train_loss.png")
    if val:
        plot_series(val, out_dir / "val_loss.png")
    if times:
        plot_series(times, out_dir / "step_time.png")
    if lrs:
        plot_series(lrs, out_dir / "learning_rate.png")


def main():
    parser = argparse.ArgumentParser(description="Genera grafici PNG semplici dalle statistiche CSV.")
    parser.add_argument("--csv", default="data/logs/training_stats.csv")
    parser.add_argument("--out_dir", default="data/plots")
    args = parser.parse_args()
    make_plots(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
