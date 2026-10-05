from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UIBugDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_size: int = 224, mode: str = "train"):
        self.csv_path = Path(csv_path)
        self.mode = mode
        self.image_size = image_size
        self.samples = self._load_csv()
        self.transform = self._build_transform()

    def _load_csv(self):
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        valid_rows = []
        for row in rows:
            if not row.get("image_path"):
                continue
            image_path = row["image_path"]
            try:
                label = int(row["label"])
            except (TypeError, ValueError):
                continue
            valid_rows.append((image_path, label))
        return valid_rows

    def _build_transform(self):
        if self.mode == "train":
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
