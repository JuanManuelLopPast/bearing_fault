
import numpy as np


def extract_time_features(signal):
    """
    Extract statistical and time-domain features
    from a one-dimensional signal.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional signal.

    Returns
    -------
    dict
        Dictionary containing extracted features.
    """

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Signal must be a non-empty 1D array.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    mean = np.mean(signal)
    std = np.std(signal)
    minimum = np.min(signal)
    maximum = np.max(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak_to_peak = maximum - minimum
    energy = np.sum(signal ** 2)

    crest_factor = (
        np.max(np.abs(signal)) / rms
        if rms > 0
        else 0.0
    )

    features = {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "energy": energy,
        "crest_factor": crest_factor,
    }

    return features
