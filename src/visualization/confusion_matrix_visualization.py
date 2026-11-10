
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from src.data_processing.config import (
    CLASS_NAMES,
    FIGURES_DIR,
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_LABELS = list(CLASS_NAMES.keys())

CLASS_TICK_LABELS = [
    CLASS_NAMES[class_id]
    for class_id in CLASS_LABELS
]


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_predictions(predictions_file, fold=None):
    """
    Load and validate model predictions.

    Parameters
    ----------
    predictions_file : str or Path
        Path to the predictions CSV file.

    fold : int, optional
        Select predictions from a specific CV fold.
        If None, use predictions from all folds.

    Returns
    -------
    pandas.DataFrame
        Validated predictions.
    """

    predictions_file = Path(predictions_file)

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}"
        )

    df = pd.read_csv(predictions_file)

    if df.empty:
        raise ValueError("The predictions file is empty.")

    required_columns = [
        "y_true",
        "y_pred",
        "fold",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in predictions file: {missing_columns}"
        )

    # Convert labels and fold identifiers to numeric values.
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Predictions contain missing or non-numeric values."
        )

    # Verify that class labels are valid.
    valid_labels = set(CLASS_LABELS)

    observed_labels = set(
        df["y_true"].unique()
    ) | set(
        df["y_pred"].unique()
    )

    invalid_labels = observed_labels - valid_labels

    if invalid_labels:
        raise ValueError(
            f"Invalid class labels found: {invalid_labels}"
        )

    # Optionally select a specific fold.
    if fold is not None:
        df = df[df["fold"] == fold].copy()

        if df.empty:
            raise ValueError(
                f"No predictions found for fold {fold}."
            )

    return df


# ============================================================
# CONFUSION MATRIX CALCULATION
# ============================================================

def calculate_confusion_matrix_values(df, normalize=False):
    """
    Calculate the confusion matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated predictions.

    normalize : bool, default=False
        Normalize each row by its corresponding true-class total.

    Returns
    -------
    numpy.ndarray
        Absolute or normalized confusion matrix.
    """

    cm = confusion_matrix(
        df["y_true"],
        df["y_pred"],
        labels=CLASS_LABELS,
    )

    if normalize:
        row_totals = cm.sum(axis=1, keepdims=True)

        cm = np.divide(
            cm.astype(float),
            row_totals,
            out=np.zeros_like(cm, dtype=float),
            where=row_totals != 0,
        )

    return cm


# ============================================================
# CONFUSION MATRIX PLOTTING
# ============================================================

def _plot_confusion_matrix(
    cm,
    normalized=False,
    title=None,
    save_path=None,
    show=False,
):
    """
    Internal function to plot a confusion matrix.

    Parameters
    ----------
    cm : numpy.ndarray
        Confusion matrix values.

    normalized : bool
        Indicates whether the matrix is normalized.

    title : str, optional
        Figure title.

    save_path : str or Path, optional
        Output path for the figure.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    # Configure axes.
    ax.set(
        xticks=np.arange(len(CLASS_LABELS)),
        yticks=np.arange(len(CLASS_LABELS)),
        xticklabels=CLASS_TICK_LABELS,
        yticklabels=CLASS_TICK_LABELS,
        xlabel="Predicted Class",
        ylabel="True Class",
    )

    if title is None:
        title = (
            "Normalized Confusion Matrix"
            if normalized
            else "Confusion Matrix"
        )

    ax.set_title(title)

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    # Annotate each matrix cell.
    threshold = cm.max() / 2 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            if normalized:
                text_value = f"{cm[i, j]:.2%}"
            else:
                text_value = f"{int(cm[i, j])}"

            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()

    # Save figure.
    if save_path is None:
        filename = (
            "normalized_confusion_matrix.png"
            if normalized
            else "confusion_matrix.png"
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


# ============================================================
# ABSOLUTE CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    predictions_file,
    fold=None,
    save_path=None,
    show=False,
):
    """
    Generate an absolute confusion matrix.

    Parameters
    ----------
    predictions_file : str or Path
        Path to the predictions CSV file.

    fold : int, optional
        Select a specific CV fold.
        If None, use all folds.

    save_path : str or Path, optional
        Custom output path.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    df = load_predictions(
        predictions_file,
        fold=fold,
    )

    cm = calculate_confusion_matrix_values(
        df,
        normalize=False,
    )

    predictions_file = Path(predictions_file)

    title = f"Confusion Matrix - {predictions_file.stem}"

    if fold is not None:
        title += f" - Fold {fold}"

    if save_path is None:
        suffix = f"_fold_{fold}" if fold is not None else ""

        save_path = (
            FIGURES_DIR
            / f"{predictions_file.stem}_confusion_matrix{suffix}.png"
        )

    return _plot_confusion_matrix(
        cm,
        normalized=False,
        title=title,
        save_path=save_path,
        show=show,
    )


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

def plot_normalized_confusion_matrix(
    predictions_file,
    fold=None,
    save_path=None,
    show=False,
):
    """
    Generate a row-normalized confusion matrix.

    Each row represents the distribution of predictions
    for one true class.

    Parameters
    ----------
    predictions_file : str or Path
        Path to the predictions CSV file.

    fold : int, optional
        Select a specific CV fold.
        If None, use all folds.

    save_path : str or Path, optional
        Custom output path.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    df = load_predictions(
        predictions_file,
        fold=fold,
    )

    cm = calculate_confusion_matrix_values(
        df,
        normalize=True,
    )

    predictions_file = Path(predictions_file)

    title = (
        f"Normalized Confusion Matrix - {predictions_file.stem}"
    )

    if fold is not None:
        title += f" - Fold {fold}"

    if save_path is None:
        suffix = f"_fold_{fold}" if fold is not None else ""

        save_path = (
            FIGURES_DIR
            / f"{predictions_file.stem}_normalized_confusion_matrix{suffix}.png"
        )

    return _plot_confusion_matrix(
        cm,
        normalized=True,
        title=title,
        save_path=save_path,
        show=show,
    )
