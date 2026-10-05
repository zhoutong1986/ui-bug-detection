from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image


def build_parser():
    parser = argparse.ArgumentParser(description="Collect UI screenshots")
    parser.add_argument("--output", type=str, default="data/raw", help="Directory to save screenshots")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of screenshots to save")
    parser.add_argument("--width", type=int, default=1440,
                        help="Target screenshot width")
    parser.add_argument("--height", type=int, default=900,
                        help="Target screenshot height")
    return parser


def create_placeholder_ui_image(path: Path, width: int, height: int):
    image = Image.new("RGB", (width, height), color=(245, 245, 245))
    # Simulate a simple UI layout for testing before real browser screenshots are collected
    # Header
    for x in range(0, width, 2):
        for y in range(0, 80, 2):
            image.putpixel((x, y), (220, 220, 220))
    # Sidebar
    for x in range(0, 220, 2):
        for y in range(80, height, 2):
            image.putpixel((x, y), (230, 230, 230))
    # Buttons
    for x in range(260, width - 260, 120):
        for y in range(120, 260, 40):
            for dx in range(0, 90):
                for dy in range(0, 32):
                    if 0 <= x + dx < width and 0 <= y + dy < height:
                        image.putpixel((x + dx, y + dy), (180, 210, 255))
    # Text blocks
    for x in range(260, width - 100, 3):
        for y in range(320, 760, 22):
            for dx in range(0, 160):
                if 0 <= x + dx < width and 0 <= y < height:
                    image.putpixel((x + dx, y), (200, 200, 200))
    image.save(path)


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        image_path = output_dir / f"ui_{i:05d}.png"
        create_placeholder_ui_image(image_path, args.width, args.height)
        print(f"Saved {image_path}")

    print(f"Collected {args.count} sample screenshots in {output_dir}")


if __name__ == "__main__":
    main()
