from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from config import LABELS


def build_parser():
    parser = argparse.ArgumentParser(description="Make a simple annotation CSV from collected screenshots")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory containing screenshots")
    parser.add_argument("--output", type=str, default="data/annotations.csv", help="Annotation CSV file path")
    return parser


def collect_images(root: str | Path):
    root = Path(root)
    images = []
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            images.append(str(file.resolve()))
    return images


def main():
    parser = build_parser()
    args = parser.parse_args()

    image_paths = collect_images(args.input_dir)
    rows = []
    for path in image_paths:
        label = 0
        label_name = LABELS.get(label, "normal")
        rows.append({
            "image_path": path,
            "label": label,
            "label_name": label_name,
            "source": "manual",
            "notes": "" 
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "label_name", "source", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} placeholder annotation entries at {output_path}")
    print("Tip: update labels manually to match your UI bug categories.")


if __name__ == "__main__":
    main()
