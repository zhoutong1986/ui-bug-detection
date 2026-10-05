from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_PATH = DATA_DIR / "annotations.csv"
RESULTS_DIR = Path("results")
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"

LABELS = {
    0: "normal",
    1: "layout_shift",
    2: "overlap",
    3: "text_cutoff",
    4: "button_hidden",
    5: "missing_element",
    6: "color_issue",
    7: "spacing_error",
}

CLASS_NAMES = list(LABELS.values())
NUM_CLASSES = len(LABELS)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42
