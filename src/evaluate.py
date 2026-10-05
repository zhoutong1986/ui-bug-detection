from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from config import BATCH_SIZE, EPOCHS, IMAGE_SIZE, LABELS, LEARNING_RATE, NUM_CLASSES, SEED
from dataset import UIBugDataset
from models import get_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train UI bug recognition model")
    parser.add_argument("--csv", type=str, default="data/annotations.csv", help="CSV annotation file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/checkpoints")
    return parser.parse_args()


def train():
    args = parse_args()
    set_seed(SEED)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}\nPlease create annotations first.")

    dataset = UIBugDataset(csv_path=csv_path, image_size=args.image_size, mode="train")
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(num_classes=NUM_CLASSES, backbone=args.backbone).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        train_acc = accuracy_score(all_labels, all_preds)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, args.device)

        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={running_loss / len(train_loader.dataset):.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), acc, f1


if __name__ == "__main__":
    train()
