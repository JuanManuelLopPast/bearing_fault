
import pandas as pd
from pathlib import Path

from src.data_processing.config import SEGMENT_LENGTH
from src.data_processing.dataset_config import (
    get_bearing_class,
    SELECTED_BEARINGS
)
from src.data_processing.data_loader import (
    load_mat_file,
    get_signal_info
)
from src.data_processing.segmentation import segment_signals

from src.feature_extraction.time_features import extract_time_features
from src.feature_extraction.frequency_features import extract_frequency_features
from src.feature_extraction.features import combine_features


def find_mat_files(data_dir):

    data_dir = Path(data_dir)
    mat_files = sorted(data_dir.rglob("*.mat"))

    return mat_files


def process_experiment(file_path, fs, window_duration):

    file_path = Path(file_path)

    experiment_name = file_path.stem
    bearing_id = file_path.parent.name
    experiment_id = file_path.stem

    class_info = get_bearing_class(bearing_id)

    data = load_mat_file(file_path)

    current = get_signal_info(
        data,
        experiment_name,
        "phase_current_1"
    )

    vibration = get_signal_info(
        data,
        experiment_name,
        "vibration_1"
    )

    segment_length = int(fs * window_duration)

    segments = segment_signals(
        {
            "current": current["data"],
            "vibration": vibration["data"]
        },
        segment_length=segment_length
    )

    current_segments = segments["current"]
    vibration_segments = segments["vibration"]

    num_segments = len(current_segments)

    rows = []

    for i in range(num_segments):

        current_time_features = extract_time_features(
            current_segments[i]
        )

        vibration_time_features = extract_time_features(
            vibration_segments[i]
        )

        current_frequency_features = extract_frequency_features(
            current_segments[i],
            fs,
            apply_window=False
        )

        vibration_frequency_features = extract_frequency_features(
            vibration_segments[i],
            fs,
            apply_window=False
        )

        features = combine_features(
            current_time_features,
            vibration_time_features,
            current_frequency_features,
            vibration_frequency_features
        )

        features["bearing_id"] = bearing_id
        features["class_id"] = class_info["class_id"]
        features["class_name"] = class_info["class_name"]
        features["experiment_id"] = experiment_id
        features["segment_id"] = i

        rows.append(features)

    return rows


def build_dataset(data_dir, fs, window_duration):

    mat_files = find_mat_files(data_dir)

    all_rows = []

    for file_path in mat_files:

        bearing_id = file_path.parent.name

        if bearing_id not in SELECTED_BEARINGS:
            continue

        print(f"Processing: {file_path}")

        rows = process_experiment(
            file_path,
            fs,
            window_duration
        )

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    return df
