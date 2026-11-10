
import numpy as np


def compute_fft(signal, fs, apply_window=False):
    """
    Compute the one-sided FFT magnitude spectrum.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    apply_window : bool
        Apply a Hann window before FFT.

    Returns
    -------
    frequencies : ndarray
        Frequency vector.
    magnitude : ndarray
        FFT magnitude.
    """

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Signal must be a non-empty 1D array.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    n = len(signal)

    # Remove the DC component.
    signal = signal - np.mean(signal)

    # Apply Hann window if requested.
    if apply_window:
        window = np.hanning(n)
        signal = signal * window

    # Compute one-sided FFT.
    spectrum = np.fft.rfft(signal)
    magnitude = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(n, d=1 / fs)

    return frequencies, magnitude


def extract_frequency_features(signal, fs, apply_window=False):
    """
    Extract frequency-domain features.

    Returns
    -------
    dict
        Dominant magnitude, spectral energy and spectral centroid.
    """

    frequencies, magnitude = compute_fft(
        signal,
        fs,
        apply_window=apply_window
    )

    # Exclude the DC component.
    frequencies_no_dc = frequencies[1:]
    magnitude_no_dc = magnitude[1:]

    if magnitude_no_dc.size == 0:
        raise ValueError("Signal is too short to calculate frequency features.")

    dominant_index = np.argmax(magnitude_no_dc)

    dominant_magnitude = magnitude_no_dc[dominant_index]
    spectral_energy = np.sum(magnitude_no_dc ** 2)
    magnitude_sum = np.sum(magnitude_no_dc)

    if magnitude_sum > 0:
        spectral_centroid = (
            np.sum(frequencies_no_dc * magnitude_no_dc)
            / magnitude_sum
        )
    else:
        spectral_centroid = 0.0

    features = {
        "dominant_magnitude": float(dominant_magnitude),
        "spectral_energy": float(spectral_energy),
        "spectral_centroid": float(spectral_centroid),
    }

    return features
