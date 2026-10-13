
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from src.config import SEGMENT_LENGTH


def _decode_name(value) -> str:
    """Convert a MATLAB character field into a Python string."""
    array = np.asarray(value).squeeze()

    if isinstance(array, bytes):
        return array.decode("utf-8").strip()

    if isinstance(array, str):
        return array.strip()

    if array.dtype.kind in {"U", "S"}:
        return "".join(array.astype(str).ravel()).strip()

    if array.dtype == object and array.size == 1:
        return _decode_name(array.item())

    return str(array).strip()


def load_signals(
    mat_path: str | Path,
    signal_names: tuple[str, ...] = (
        "phase_current_1",
        "vibration_1",
    ),
) -> dict[str, np.ndarray]:
    """Load selected signals from a Paderborn MATLAB file."""
    mat_path = Path(mat_path)

    if not mat_path.is_file():
        raise FileNotFoundError(f"MATLAB file not found: {mat_path}")

    mat = loadmat(mat_path)
    experiment_name = mat_path.stem

    if experiment_name not in mat:
        raise KeyError(
            f"Expected MATLAB structure '{experiment_name}' "
            f"was not found in {mat_path.name}"
        )

    experiment = mat[experiment_name]
    records = experiment["Y"][0, 0]
    signals = {}

    for record in records.ravel():
        name = _decode_name(record["Name"])

        if name not in signal_names:
            continue

        values = np.asarray(record["Data"]).squeeze()

        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"Invalid signal data for '{name}'")

        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"Signal '{name}' is not numeric")

        signals[name] = values.astype(np.float64, copy=False)

    missing = set(signal_names) - signals.keys()
    if missing:
        raise KeyError(f"Missing signals in {mat_path.name}: {sorted(missing)}")

    return signals


def segment_signals(
    signals: dict[str, np.ndarray],
    segment_length: int = SEGMENT_LENGTH,
) -> dict[str, np.ndarray]:
    """Split signals into aligned, non-overlapping segments."""
    if not signals:
        raise ValueError("At least one signal is required.")

    if segment_length <= 0:
        raise ValueError("segment_length must be positive.")

    if any(np.asarray(x).ndim != 1 for x in signals.values()):
        raise ValueError("All signals must be one-dimensional.")

    common_length = min(len(x) for x in signals.values())
    n_segments = common_length // segment_length

    if n_segments == 0:
        raise ValueError("Signals are shorter than one complete segment.")

    usable_length = n_segments * segment_length

    return {
        name: np.asarray(values[:usable_length]).reshape(
            n_segments, segment_length
        )
        for name, values in signals.items()
    }
