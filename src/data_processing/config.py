
from pathlib import Path

# Project directories
ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = ROOT_DIR / "data" / "raw" / "paderborn"
FEATURES_DIR = ROOT_DIR / "data" / "processed" / "features"
FEATURES_FILE = FEATURES_DIR / "features.csv"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FOLDS_DIR = METRICS_DIR / "folds"
FIGURES_DIR = RESULTS_DIR / "figures"

# Classification
CLASS_NAMES = {
    0: "Healthy",
    1: "Outer_Race",
    2: "Inner_Race",
}

# Signal processing
SAMPLING_RATE = 64_000
SEGMENT_DURATION = 0.1
SEGMENT_LENGTH = int(SAMPLING_RATE * SEGMENT_DURATION)

# Cross-validation
N_SPLITS = 5
RANDOM_STATE = 42

# Dataset metadata
METADATA_COLUMNS = [
    "bearing_id",
    "class_id",
    "class_name",
    "experiment_id",
    "segment_id",
]
