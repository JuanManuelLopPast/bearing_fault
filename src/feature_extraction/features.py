
def combine_features(
    current_features,
    vibration_features,
    current_frequency_features=None,
    vibration_frequency_features=None
):
    """
    Combine temporal and frequency features from current and vibration signals.

    Parameters
    ----------
    current_features : dict
        Temporal features extracted from the current signal.

    vibration_features : dict
        Temporal features extracted from the vibration signal.

    current_frequency_features : dict, optional
        Frequency features extracted from the current signal.

    vibration_frequency_features : dict, optional
        Frequency features extracted from the vibration signal.

    Returns
    -------
    dict
        Combined feature dictionary.
    """

    features = {}

    for name, value in current_features.items():
        features[f"current_{name}"] = value

    for name, value in vibration_features.items():
        features[f"vibration_{name}"] = value

    if current_frequency_features is not None:
        for name, value in current_frequency_features.items():

            if name == "dominant_frequency":
                continue

            features[f"current_{name}"] = value

    if vibration_frequency_features is not None:
        for name, value in vibration_frequency_features.items():

            if name == "dominant_frequency":
                continue

            features[f"vibration_{name}"] = value

    return features
