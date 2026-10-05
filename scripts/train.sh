#!/usr/bin/env bash
set -e

python src/train.py --csv data/annotations.csv --batch-size 32 --epochs 30 --image-size 224 --backbone efficientnet_b0
