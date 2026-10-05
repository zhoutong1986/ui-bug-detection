from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import UIBugDataset
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UI bug detection model")
    parser.add_argument("--csv", type=str, default="data/annotations.csv", help="CSV annotation file")
    parser.add_argument("--model", type=str, default="results/checkpoints/best_model.pth", help="Trained model checkpoint")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    model = get_model(num_classes=8, backbone="efficientnet_b0")
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.to(args.device)
    model.eval()

    dataset = UIBugDataset(csv_path=csv_path, image_size=args.image_size, mode="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    preds = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(args.device)
            outputs = model(images)
            pred = outputs.argmax(dim=1).cpu().tolist()
            preds.extend(pred)
            labels.extend(batch_labels.tolist())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(labels, preds, zero_division=0))
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
