
from datetime import datetime

import pandas as pd

from src.data_processing.config import (
    RESULTS_DIR,
    METRICS_DIR,
    FOLDS_DIR
)


EXPERIMENTS_FILE = RESULTS_DIR / "experiments.csv"
CONFIGS_DIR = RESULTS_DIR / "configs"


def save_fold_results(results, model_name, feature_set):
    """
    Save cross-validation results for each fold.
    """

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_{feature_set}_folds.csv"
    output_path = METRICS_DIR / filename

    results.to_csv(output_path, index=False)

    print("\nFold results saved to:")
    print(output_path)

    return output_path


def create_summary(results, model_name, feature_set):
    """
    Create a summary containing mean and standard deviation.
    """

    summary = {}

    numeric_columns = results.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        summary[f"{column}_mean"] = results[column].mean()
        summary[f"{column}_std"] = results[column].std()

    summary["model"] = model_name
    summary["feature_set"] = feature_set

    return pd.DataFrame([summary])


def save_experiment_results(results, model_name, feature_set):
    """
    Save fold results and statistical summary.
    """

    save_fold_results(
        results,
        model_name,
        feature_set
    )

    summary = create_summary(
        results,
        model_name,
        feature_set
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = (
        METRICS_DIR
        / f"{model_name}_{feature_set}_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\nSummary saved to:")
    print(summary_path)

    return summary


def create_experiment_config(
    model_name,
    feature_set,
    n_splits,
    random_state,
    fs,
    window_duration
):
    """
    Create the configuration of an experiment.
    """

    config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "feature_set": feature_set,
        "n_splits": n_splits,
        "random_state": random_state,
        "fs": fs,
        "window_duration": window_duration
    }

    return config


def save_experiment_config(config, model_name, feature_set):
    """
    Save experiment configuration.
    """

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    config_df = pd.DataFrame([config])

    output_path = (
        CONFIGS_DIR
        / f"{model_name}_{feature_set}_config.csv"
    )

    config_df.to_csv(output_path, index=False)

    print("\nConfiguration saved to:")
    print(output_path)

    return output_path


def get_next_experiment_id():
    """
    Generate the next experiment identifier.
    """

    if not EXPERIMENTS_FILE.exists():
        return "EXP_0001"

    experiments = pd.read_csv(EXPERIMENTS_FILE)

    if experiments.empty:
        return "EXP_0001"

    last_id = experiments["experiment_id"].iloc[-1]

    number = int(last_id.split("_")[1])

    return f"EXP_{number + 1:04d}"


def register_experiment(
    experiment_id,
    model_name,
    feature_set,
    fold_results,
    n_splits,
    random_state,
    fs,
    window_duration
):
    """
    Register an experiment in the master file
    and save its individual fold results.
    """

    results = pd.DataFrame(fold_results)

    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]

    experiment = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "feature_set": feature_set,
        "n_splits": n_splits,
        "random_state": random_state,
        "fs": fs,
        "window_duration": window_duration
    }

    for metric in metric_names:
        experiment[f"{metric}_mean"] = results[metric].mean()
        experiment[f"{metric}_std"] = results[metric].std()

    new_row = pd.DataFrame([experiment])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)

    if EXPERIMENTS_FILE.exists():
        experiments = pd.read_csv(EXPERIMENTS_FILE)

        experiments = pd.concat(
            [experiments, new_row],
            ignore_index=True
        )
    else:
        experiments = new_row

    experiments.to_csv(
        EXPERIMENTS_FILE,
        index=False
    )

    fold_file = FOLDS_DIR / f"{experiment_id}.csv"

    results.to_csv(
        fold_file,
        index=False
    )

    print("\nExperiment registered:")
    print(experiment_id)

    print("\nMaster results:")
    print(EXPERIMENTS_FILE)

    print("\nFold results:")
    print(fold_file)

    return experiment_id
