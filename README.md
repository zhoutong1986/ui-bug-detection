# UI Bug Detection Project

This project is a starter template for training a deep learning model to detect UI bugs from screenshots.

## Overview

- Inputs: UI screenshots
- Task: binary or multi-class classification
- Labels: `normal` vs bug categories
- Model: EfficientNet / ResNet based image classifier
- Goal: recognize UI anomalies from visual patterns

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --config configs/training.yaml
```

## Data format

The dataset should be organized as a CSV file with columns:

```csv
image_path,label
path/to/image1.png,0
path/to/image2.png,1
```

Labels:
- 0: normal
- 1: layout_shift
- 2: overlap
- 3: text_cutoff
- 4: button_hidden
- 5: missing_element
- 6: color_issue
- 7: spacing_error

## File structure

- `src/collector.py`: collect screenshot data
- `src/dataset.py`: dataset loading and transforms
- `src/models.py`: model definition
- `src/train.py`: training loop
- `src/evaluate.py`: evaluation
- `src/inference.py`: infer on new images
