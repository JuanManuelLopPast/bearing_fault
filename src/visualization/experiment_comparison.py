
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing.config import (
    RESULTS_DIR,
    FIGURES_DIR,
)


# ============================================================
# CONFIGURATION
# ============================================================

METRICS = {
    "accuracy": "Accuracy",
    "precision_macro": "Macro Precision",
    "recall_macro": "Macro Recall",
    "f1_macro": "Macro F1-score",
}


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_experiments(experiments_file=None):
    """
    Load and validate the experiment registry.

    Parameters
    ----------
    experiments_file : str or Path, optional
        Path to experiments.csv.

    Returns
    -------
    pandas.DataFrame
        Validated experiment registry.
    """

    if experiments_file is None:
        experiments_file = RESULTS_DIR / "experiments.csv"

    experiments_file = Path(experiments_file)

    if not experiments_file.exists():
        raise FileNotFoundError(
            f"Experiment registry not found: {experiments_file}"
        )

    df = pd.read_csv(experiments_file)

    if df.empty:
        raise ValueError(
            "The experiment registry is empty."
        )

    required_columns = [
        "experiment_id",
        "model",
        "feature_set",
    ]

    for metric in METRICS:
        required_columns.extend([
            f"{metric}_mean",
            f"{metric}_std",
        ])

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in experiment registry: {missing_columns}"
        )

    # Validate experiment identifiers.
    if df["experiment_id"].isna().any():
        raise ValueError(
            "Experiment identifiers cannot contain missing values."
        )

    if df["experiment_id"].duplicated().any():
        raise ValueError(
            "Duplicate experiment identifiers found."
        )

    # Normalize text fields.
    for column in ["experiment_id", "model", "feature_set"]:
        df[column] = df[column].astype(str).str.strip()

        if (df[column] == "").any():
            raise ValueError(
                f"Empty values found in column: {column}"
            )

    # Convert metrics to numeric values.
    for metric in METRICS:

        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"

        df[mean_column] = pd.to_numeric(
            df[mean_column],
            errors="coerce",
        )

        df[std_column] = pd.to_numeric(
            df[std_column],
            errors="coerce",
        )

    metric_columns = [
        column
        for metric in METRICS
        for column in [
            f"{metric}_mean",
            f"{metric}_std",
        ]
    ]

    if not np.isfinite(
        df[metric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Experiment metrics contain missing or non-finite values."
        )

    # Validate metric ranges.
    for metric in METRICS:

        means = df[f"{metric}_mean"]
        stds = df[f"{metric}_std"]

        if not means.between(0, 1).all():
            raise ValueError(
                f"{metric}_mean values must be between 0 and 1."
            )

        if (stds < 0).any():
            raise ValueError(
                f"{metric}_std values cannot be negative."
            )

    return df


# ============================================================
# EXPERIMENT SELECTION
# ============================================================

def select_experiments(df, experiment_ids=None):
    """
    Select experiments from the registry.

    Parameters
    ----------
    df : pandas.DataFrame
        Experiment registry.

    experiment_ids : list of str, optional
        Experiment identifiers to include.
        If None, include all experiments.

    Returns
    -------
    pandas.DataFrame
        Selected experiments.
    """

    if experiment_ids is None:
        return df.copy()

    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    if not experiment_ids:
        raise ValueError(
            "At least one experiment ID must be provided."
        )

    experiment_ids = [
        str(experiment_id).strip()
        for experiment_id in experiment_ids
    ]

    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError(
            "Duplicate experiment IDs were provided."
        )

    available_ids = set(df["experiment_id"])

    missing_ids = set(experiment_ids) - available_ids

    if missing_ids:
        raise ValueError(
            f"Experiments not found in registry: {missing_ids}"
        )

    selected_df = df[
        df["experiment_id"].isin(experiment_ids)
    ].copy()

    # Preserve the order specified by the user.
    selected_df["_order"] = pd.Categorical(
        selected_df["experiment_id"],
        categories=experiment_ids,
        ordered=True,
    )

    selected_df = (
        selected_df
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return selected_df


# ============================================================
# EXPERIMENT COMPARISON
# ============================================================

def plot_experiment_comparison(
    experiments_file=None,
    experiment_ids=None,
    save_path=None,
    show=False,
):
    """
    Compare multiple experiments across all evaluation metrics.

    Each metric is represented by a group of bars.
    Each bar represents one experiment.

    Error bars represent the standard deviation
    across cross-validation folds.

    Parameters
    ----------
    experiments_file : str or Path, optional
        Path to experiments.csv.

    experiment_ids : list of str, optional
        Experiments to compare.
        If None, compare all registered experiments.

    save_path : str or Path, optional
        Custom output path for the figure.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    # Load and select experiments.
    df = load_experiments(experiments_file)

    df = select_experiments(
        df,
        experiment_ids=experiment_ids,
    )

    if df.empty:
        raise ValueError(
            "No experiments available for comparison."
        )

    # Create readable experiment labels.
    experiment_labels = [
        (
            f"{row.model}\n"
            f"{row.feature_set}\n"
            f"{row.experiment_id}"
        )
        for row in df.itertuples()
    ]

    n_experiments = len(df)
    n_metrics = len(METRICS)

    # Configure grouped bars.
    x_positions = np.arange(n_metrics)

    group_width = 0.8
    bar_width = group_width / n_experiments

    offsets = (
        np.arange(n_experiments)
        - (n_experiments - 1) / 2
    ) * bar_width

    fig, ax = plt.subplots(
        figsize=(max(11, n_experiments * 1.8), 7),
    )

    # Plot each experiment.
    for index, (_, row) in enumerate(df.iterrows()):

        means = [
            row[f"{metric}_mean"]
            for metric in METRICS
        ]

        stds = [
            row[f"{metric}_std"]
            for metric in METRICS
        ]

        ax.bar(
            x_positions + offsets[index],
            means,
            width=bar_width * 0.9,
            yerr=stds,
            capsize=3,
            alpha=0.85,
            label=experiment_labels[index],
        )

    # Configure axes.
    ax.set_title(
        "Experiment Performance Comparison",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Score")

    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        list(METRICS.values()),
        rotation=0,
    )

    ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    ax.legend(
        title="Experiments",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    # Save figure.
    if save_path is None:
        save_path = (
            FIGURES_DIR / "experiment_comparison.png"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

    return save_path
