
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    FEATURES_FILE,
    RAW_DATA_DIR,
    SAMPLING_RATE,
    SEGMENT_LENGTH,
)
from src.data import load_signals, segment_signals


SIGNALS = {
    "current": "phase_current_1",
    "vibration": "vibration_1",
}


def extract_features(signal: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """Extract 8 temporal and 3 frequency-domain features."""
    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("Signal must be a non-empty 1D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite values.")
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")

    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))

    features = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": rms,
        "peak_to_peak": float(np.ptp(x)),
        "energy": float(np.sum(x**2)),
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
    }

    # Remove DC before calculating the spectrum.
    centered = x - np.mean(x)
    magnitude = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(x.size, d=1 / fs)

    magnitude_sum = float(np.sum(magnitude))

    features.update({
        "dominant_magnitude": float(np.max(magnitude)),
        "spectral_energy": float(np.sum(magnitude**2)),
        "spectral_centroid": (
            float(np.sum(frequencies * magnitude) / magnitude_sum)
            if magnitude_sum > 0
            else 0.0
        ),
    })

    return features


def _class_id(bearing_id: str) -> int:
    """Map a bearing identifier to its class."""
    if bearing_id.startswith("KA"):
        return 1
    if bearing_id.startswith("KI"):
        return 2
    if bearing_id.startswith("K") and bearing_id[1:].isdigit():
        return 0

    raise ValueError(f"Unknown bearing identifier: {bearing_id}")


def build_feature_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_file: Path = FEATURES_FILE,
) -> pd.DataFrame:
    """Build and save the feature dataset from Paderborn MAT files."""
    mat_files = sorted(Path(raw_dir).rglob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No MAT files found in {raw_dir}")

    rows = []

    for mat_path in mat_files:
        bearing_id = mat_path.parent.name
        class_id = _class_id(bearing_id)

        signals = load_signals(mat_path)
        segments = segment_signals(
            signals,
            segment_length=SEGMENT_LENGTH,
        )

        for segment_id in range(len(next(iter(segments.values())))):
            row = {
                "bearing_id": bearing_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "experiment_id": mat_path.stem,
                "segment_id": segment_id,
            }

            for prefix, signal_name in SIGNALS.items():
                row.update({
                    f"{prefix}_{name}": value
                    for name, value in extract_features(
                        segments[signal_name][segment_id]
                    ).items()
                })

            rows.append(row)

    df = pd.DataFrame(rows)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df
