
import argparse

from src.data_processing.config import (
    RAW_DATA_DIR,
    FEATURES_DIR,
    FEATURES_FILE,
    SAMPLING_RATE,
    SEGMENT_DURATION
)

from src.feature_extraction.dataset_builder import build_dataset

from src.experiments.experiment_runner import (
    main as run_experiments
)


# --------------------------------------------------
# DATASET PREPARATION
# --------------------------------------------------

def prepare_dataset(force_rebuild=False):
    """
    Reuse the processed dataset if it exists.
    Otherwise, generate it from the raw Paderborn data.
    """

    print("\n--- DATASET PREPARATION ---")

    # Reuse existing dataset
    if FEATURES_FILE.exists() and not force_rebuild:

        print("Existing dataset found.")
        print(f"Dataset: {FEATURES_FILE}")
        print("Reusing processed features.")

        return FEATURES_FILE

    # Generate dataset
    if force_rebuild:
        print("Forced dataset reconstruction.")
    else:
        print("Processed dataset not found.")

    print("Generating dataset from raw MAT files...")

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = build_dataset(
        data_dir=RAW_DATA_DIR,
        fs=SAMPLING_RATE,
        window_duration=SEGMENT_DURATION
    )

    if df.empty:
        raise ValueError(
            "Dataset generation failed: no samples were produced. "
            "Check the raw data directory and selected bearings."
        )

    # Save processed dataset
    df.to_csv(
        FEATURES_FILE,
        index=False
    )

    print("\nDataset successfully generated.")
    print(f"Total samples: {len(df)}")
    print(f"Dataset dimensions: {df.shape}")
    print(f"Saved to: {FEATURES_FILE}")

    return FEATURES_FILE


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Bearing Fault Diagnosis Pipeline"
    )

    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Force reconstruction of the processed dataset."
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("BEARING FAULT DIAGNOSIS PROJECT")
    print("=" * 65)

    # 1. Prepare dataset
    prepare_dataset(
        force_rebuild=args.rebuild_dataset
    )

    # 2. Execute experiments
    print("\n--- EXPERIMENT EXECUTION ---")

    run_experiments()

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
