
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing.config import (
    METRICS_DIR,
    FIGURES_DIR,
)


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_DIR = METRICS_DIR / "training_history"

HISTORY_COLUMNS = [
    "fold",
    "epoch",
    "train_loss",
    "train_accuracy",
]


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_training_history(history_file):
    """
    Load and validate a PyTorch training history CSV file.

    Parameters
    ----------
    history_file : str or Path
        Path to the training history CSV file.

    Returns
    -------
    pandas.DataFrame
        Validated training history.
    """

    history_file = Path(history_file)

    if not history_file.exists():
        raise FileNotFoundError(
            f"Training history file not found: {history_file}"
        )

    df = pd.read_csv(history_file)

    if df.empty:
        raise ValueError(
            "The training history file is empty."
        )

    missing_columns = [
        column for column in HISTORY_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in training history: {missing_columns}"
        )

    # Convert columns to numeric values.
    for column in HISTORY_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[HISTORY_COLUMNS].isna().any().any():
        raise ValueError(
            "Training history contains missing or non-numeric values."
        )

    # Validate fold and epoch identifiers.
    if (df["fold"] <= 0).any():
        raise ValueError(
            "Fold identifiers must be positive."
        )

    if (df["epoch"] <= 0).any():
        raise ValueError(
            "Epoch identifiers must be positive."
        )

    if (df["train_loss"] < 0).any():
        raise ValueError(
            "Training loss cannot be negative."
        )

    if not df["train_accuracy"].between(0, 1).all():
        raise ValueError(
            "Training accuracy must be between 0 and 1."
        )

    # Ensure that each fold has unique epochs.
    if df.duplicated(subset=["fold", "epoch"]).any():
        raise ValueError(
            "Duplicate epoch entries found within a fold."
        )

    return df.sort_values(
        ["fold", "epoch"]
    ).reset_index(drop=True)


# ============================================================
# INTERNAL PLOTTING FUNCTION
# ============================================================

def _plot_training_curves(
    df,
    metric,
    ylabel,
    title,
    save_path,
    show=False,
):
    """
    Internal function for plotting training curves.

    Parameters
    ----------
    df : pandas.DataFrame
        Training history.
    metric : str
        Metric column to visualize.
    ylabel : str
        Y-axis label.
    title : str
        Figure title.
    save_path : str or Path
        Output path.
    show : bool
        Display the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    folds = sorted(df["fold"].unique())

    for fold in folds:

        fold_data = df[df["fold"] == fold]

        ax.plot(
            fold_data["epoch"],
            fold_data[metric],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"Fold {int(fold)}",
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    ax.grid(
        linestyle="--",
        alpha=0.4,
    )

    ax.legend(
        title="Cross-validation",
    )

    if metric == "train_accuracy":
        ax.set_ylim(0, 1.05)

    fig.tight_layout()

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


# ============================================================
# COMPLETE TRAINING HISTORY
# ============================================================

def plot_training_history(
    history_file,
    fold=None,
    save_dir=None,
    show=False,
):
    """
    Generate training loss and accuracy curves.

    Parameters
    ----------
    history_file : str or Path
        Path to the training history CSV file.

    fold : int, optional
        Specific fold to visualize.
        If None, visualize all folds.

    save_dir : str or Path, optional
        Directory where figures will be saved.

    show : bool, default=False
        Display the figures.

    Returns
    -------
    dict
        Paths to the generated figures.
    """

    df = load_training_history(history_file)

    if fold is not None:

        df = df[df["fold"] == fold].copy()

        if df.empty:
            raise ValueError(
                f"No training history found for fold {fold}."
            )

    if save_dir is None:
        save_dir = FIGURES_DIR

    save_dir = Path(save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_file = Path(history_file)

    fold_suffix = (
        f"_fold_{fold}"
        if fold is not None
        else "_all_folds"
    )

    # Training loss.
    loss_path = save_dir / (
        f"{history_file.stem}_loss{fold_suffix}.png"
    )

    loss_path = _plot_training_curves(
        df=df,
        metric="train_loss",
        ylabel="Training Loss",
        title="Training Loss Across Epochs",
        save_path=loss_path,
        show=show,
    )

    # Training accuracy.
    accuracy_path = save_dir / (
        f"{history_file.stem}_accuracy{fold_suffix}.png"
    )

    accuracy_path = _plot_training_curves(
        df=df,
        metric="train_accuracy",
        ylabel="Training Accuracy",
        title="Training Accuracy Across Epochs",
        save_path=accuracy_path,
        show=show,
    )

    return {
        "loss": loss_path,
        "accuracy": accuracy_path,
    }


# ============================================================
# TRAINING LOSS ONLY
# ============================================================

def plot_training_loss(
    history_file,
    fold=None,
    save_path=None,
    show=False,
):
    """
    Generate a training loss curve.

    Parameters
    ----------
    history_file : str or Path
        Path to the training history CSV file.

    fold : int, optional
        Specific fold to visualize.
        If None, visualize all folds.

    save_path : str or Path, optional
        Custom output path.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    df = load_training_history(history_file)

    if fold is not None:

        df = df[df["fold"] == fold].copy()

        if df.empty:
            raise ValueError(
                f"No training history found for fold {fold}."
            )

    if save_path is None:

        history_file = Path(history_file)

        suffix = (
            f"_fold_{fold}"
            if fold is not None
            else "_all_folds"
        )

        save_path = (
            FIGURES_DIR
            / f"{history_file.stem}_loss{suffix}.png"
        )

    return _plot_training_curves(
        df=df,
        metric="train_loss",
        ylabel="Training Loss",
        title="Training Loss Across Epochs",
        save_path=save_path,
        show=show,
    )
