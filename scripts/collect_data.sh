#!/usr/bin/env bash
set -e

python src/collector.py --output data/raw --count 1000 --width 1440 --height 900
python src/annotator.py --input-dir data/raw --output data/annotations.csv
