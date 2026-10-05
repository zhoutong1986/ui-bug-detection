from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from PIL import Image

from config import LABELS


def read_annotations(csv_path: str | Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_annotations(rows, csv_path: str | Path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "label_name", "source", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def ensure_valid_image(image_path: str | Path):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(path) as img:
        img.verify()
    return True


def label_to_name(label):
    return LABELS.get(int(label), "unknown")


def parse_args():
    parser = argparse.ArgumentParser(description="UI bug detection project")
    parser.add_argument("--csv", type=str, default="data/annotations.csv", help="CSV annotation file")
    return parser.parse_args()
