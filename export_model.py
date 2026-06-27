import argparse
import json
import shutil
import struct
from pathlib import Path


DTYPES = {
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.int64": "I64",
    "torch.int32": "I32",
    "torch.int8": "I8",
    "torch.uint8": "U8",
    "torch.bool": "BOOL",
}


def tensor_bytes(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return tensor.numpy().tobytes()


def save_safetensors(state_dict, path):
    metadata = {}
    data_chunks = []
    offset = 0
    for name, tensor in state_dict.items():
        raw = tensor_bytes(tensor)
        dtype = DTYPES.get(str(tensor.dtype))
        if dtype is None:
            raise ValueError(f"dtype non supportato per {name}: {tensor.dtype}")
        metadata[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_chunks.append(raw)
        offset += len(raw)

    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for chunk in data_chunks:
            f.write(chunk)


def main():
    parser = argparse.ArgumentParser(description="Esporta modello, tokenizer e config in export/.")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--quantized_checkpoint", default=None)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--out_dir", default="export")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("FAIL: PyTorch non installato")
        return 1

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = out_dir / "weights"
    tokenizer_dir = out_dir / "tokenizer"
    weights_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    save_safetensors(checkpoint["model"], weights_dir / "model.safetensors")
    shutil.copyfile(args.checkpoint, weights_dir / "model.pt")
    if args.quantized_checkpoint:
        shutil.copyfile(args.quantized_checkpoint, weights_dir / "model_quantized.pt")
    (out_dir / "config.json").write_text(json.dumps(checkpoint["config"], indent=2), encoding="utf-8")
    shutil.copyfile(args.tokenizer, tokenizer_dir / "tokenizer.json")
    manifest = {
        "format": "mini_llm_export",
        "weights": "weights/model.safetensors",
        "torch_checkpoint": "weights/model.pt",
        "quantized_checkpoint": "weights/model_quantized.pt" if args.quantized_checkpoint else None,
        "config": "config.json",
        "tokenizer": "tokenizer/tokenizer.json",
        "source_checkpoint": args.checkpoint,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README_export.md").write_text(
        "# mini_llm export\n\n"
        "Contenuto:\n\n"
        "- `weights/model.safetensors`: pesi in formato safetensors minimale\n"
        "- `weights/model.pt`: checkpoint PyTorch originale\n"
        "- `weights/model_quantized.pt`: checkpoint quantizzato, se esportato\n"
        "- `tokenizer/tokenizer.json`: tokenizer BPE\n"
        "- `config.json`: configurazione modello\n"
        "- `manifest.json`: indice dei file\n\n"
        "Uso rapido:\n\n"
        "```bash\n"
        "python run_generate.py --checkpoint export/weights/model.pt --tokenizer export/tokenizer/tokenizer.json\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"export salvato in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
