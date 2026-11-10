
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_processing.config import RESULTS_DIR, FIGURES_DIR


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

def load_experiment_registry(experiments_file=None):
    """
    Load the master experiment registry.

    Parameters
    ----------
    experiments_file : str or Path, optional
        Path to experiments.csv.

    Returns
    -------
    pandas.DataFrame
        Experiment registry.
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
        raise ValueError("The experiment registry is empty.")

    required_columns = [
        "experiment_id",
        "model",
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

    return df


# ============================================================
# EXPERIMENT METRICS COMPARISON
# ============================================================

def plot_experiment_metrics(
    experiments_file=None,
    save_path=None,
    show=False,
):
    """
    Compare the average performance metrics of all experiments.

    Each metric is displayed in a separate subplot.
    Error bars represent the standard deviation across folds.

    Parameters
    ----------
    experiments_file : str or Path, optional
        Path to experiments.csv.

    save_path : str or Path, optional
        Custom output path for the figure.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    df = load_experiment_registry(experiments_file)

    # Create a readable label for each experiment.
    df["experiment_label"] = (
        df["model"].astype(str).str.strip()
        + "\n"
        + df["experiment_id"].astype(str)
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 9),
    )

    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, METRICS.items()):

        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"

        means = pd.to_numeric(
            df[mean_column],
            errors="coerce",
        )

        stds = pd.to_numeric(
            df[std_column],
            errors="coerce",
        )

        valid = means.notna() & stds.notna()

        if not valid.any():
            ax.set_visible(False)
            continue

        labels = df.loc[valid, "experiment_label"]

        ax.bar(
            range(len(labels)),
            means[valid],
            yerr=stds[valid],
            capsize=5,
            alpha=0.85,
        )

        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            labels,
            rotation=0,
            fontsize=8,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )

    fig.suptitle(
        "Model Performance Comparison",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout()

    # Save figure.
    if save_path is None:
        save_path = (
            FIGURES_DIR / "experiment_metrics_comparison.png"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

    return save_path


# ============================================================
# FOLD METRICS VISUALIZATION
# ============================================================

def plot_fold_metrics(
    fold_results_file,
    metric="accuracy",
    save_path=None,
    show=False,
):
    """
    Visualize the performance of a model across CV folds.

    Parameters
    ----------
    fold_results_file : str or Path
        Path to the CSV file containing fold results.

    metric : str, default="accuracy"
        Metric to visualize.
        Available options: accuracy, precision_macro,
        recall_macro, f1_macro.

    save_path : str or Path, optional
        Custom output path for the figure.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric: {metric}. "
            f"Available metrics: {list(METRICS.keys())}"
        )

    fold_results_file = Path(fold_results_file)

    if not fold_results_file.exists():
        raise FileNotFoundError(
            f"Fold results file not found: {fold_results_file}"
        )

    df = pd.read_csv(fold_results_file)

    required_columns = ["fold", metric]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in fold results: {missing_columns}"
        )

    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce",
    )

    df = df.dropna(subset=["fold", metric]).copy()

    if df.empty:
        raise ValueError(
            f"No valid values found for metric: {metric}"
        )

    df = df.sort_values("fold")

    # Calculate descriptive statistics.
    mean_value = df[metric].mean()
    std_value = df[metric].std()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df["fold"],
        df[metric],
        marker="o",
        linewidth=2,
        markersize=7,
        label=METRICS[metric],
    )

    # Average performance across folds.
    ax.axhline(
        mean_value,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_value:.3f}",
    )

    ax.set_title(
        f"{METRICS[metric]} Across Cross-Validation Folds"
    )

    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    ax.set_xticks(df["fold"])

    ax.grid(
        linestyle="--",
        alpha=0.4,
    )

    ax.legend()

    ax.text(
        0.98,
        0.95,
        f"Mean: {mean_value:.3f}\nStd: {std_value:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )

    fig.tight_layout()

    # Generate default output filename.
    if save_path is None:
        filename = (
            f"{fold_results_file.stem}_{metric}_folds.png"
        )
        save_path = FIGURES_DIR / filename

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

    return save_path
