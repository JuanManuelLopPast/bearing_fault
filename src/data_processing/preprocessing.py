
import numpy as np
import pandas as pd
from src.data_processing.config import METADATA_COLUMNS


# Temporal features: 8 per signal
TIME_FEATURE_COLUMNS = [
    "current_mean",
    "current_std",
    "current_min",
    "current_max",
    "current_rms",
    "current_peak_to_peak",
    "current_energy",
    "current_crest_factor",
    "vibration_mean",
    "vibration_std",
    "vibration_min",
    "vibration_max",
    "vibration_rms",
    "vibration_peak_to_peak",
    "vibration_energy",
    "vibration_crest_factor",
]

# Frequency-domain features: 3 per signal
FREQUENCY_ONLY_FEATURE_COLUMNS = [
    "current_dominant_magnitude",
    "current_spectral_energy",
    "current_spectral_centroid",
    "vibration_dominant_magnitude",
    "vibration_spectral_energy",
    "vibration_spectral_centroid",
]

TIME_FREQUENCY_FEATURE_COLUMNS = (
    TIME_FEATURE_COLUMNS + FREQUENCY_ONLY_FEATURE_COLUMNS
)

FEATURE_COLUMNS = TIME_FREQUENCY_FEATURE_COLUMNS


def load_feature_dataset(file_path):
    """Load the feature dataset from a CSV file."""
    return pd.read_csv(file_path)


def check_missing_values(df):
    """Check for missing values in the dataset."""
    missing = df.isna().sum()
    total_missing = missing.sum()

    print("\nMissing Values")
    print("=" * 50)
    print(missing[missing > 0])
    print(f"\nTotal missing values: {total_missing}")

    return missing


def check_infinite_values(df):
    """Check for infinite values in numerical columns."""
    numeric_df = df.select_dtypes(include=np.number)
    infinite_count = np.isinf(numeric_df).sum()
    total_infinite = infinite_count.sum()

    print("\nInfinite Values")
    print("=" * 50)
    print(infinite_count[infinite_count > 0])
    print(f"\nTotal infinite values: {total_infinite}")

    return infinite_count


def check_constant_features(df):
    """Check whether any feature has zero variance."""
    feature_data = df[FEATURE_COLUMNS]
    unique_values = feature_data.nunique()
    constant_features = unique_values[unique_values <= 1]

    print("\nConstant Features")
    print("=" * 50)

    if constant_features.empty:
        print("No constant features found.")
    else:
        print(constant_features)

    return constant_features


def check_bearing_classes(df):
    """Verify that each bearing belongs to only one class."""
    bearing_classes = df.groupby("bearing_id")["class_name"].nunique()
    inconsistent = bearing_classes[bearing_classes > 1]

    print("\nBearing-Class Consistency")
    print("=" * 50)

    if inconsistent.empty:
        print("All bearings belong to exactly one class.")
    else:
        print("Inconsistent bearings:")
        print(inconsistent)

    return bearing_classes


def check_class_distribution(df):
    """Count the samples belonging to each class."""
    distribution = df["class_name"].value_counts().sort_index()

    print("\nClass Distribution")
    print("=" * 50)
    print(distribution)

    return distribution


def validate_dataset(df):
    """Run all dataset validation checks."""
    print("\nDATASET VALIDATION")
    print("=" * 60)

    check_missing_values(df)
    check_infinite_values(df)
    check_constant_features(df)
    check_bearing_classes(df)
    check_class_distribution(df)

    print("\nValidation completed.")


def get_feature_columns(feature_set):
    """Return the columns corresponding to a selected feature set."""
    feature_sets = {
        "time": TIME_FEATURE_COLUMNS,
        "frequency": FREQUENCY_ONLY_FEATURE_COLUMNS,
        "time_frequency": TIME_FREQUENCY_FEATURE_COLUMNS,
    }

    if feature_set not in feature_sets:
        available = ", ".join(feature_sets)

        raise ValueError(
            f"Unknown feature set: '{feature_set}'. "
            f"Available feature sets: {available}"
        )

    return feature_sets[feature_set].copy()
