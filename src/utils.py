import os
from pathlib import Path


def ensure_dir(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(directory: str | Path):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(str(p) for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})


def file_stem(path: str | Path):
    return Path(path).stem


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
