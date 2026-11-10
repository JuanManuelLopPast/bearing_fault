
import numpy as np
from src.data_processing.config import SEGMENT_LENGTH


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
