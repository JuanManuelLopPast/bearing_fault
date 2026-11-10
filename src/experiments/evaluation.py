
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data_processing.config import CLASS_NAMES


CLASS_LABELS = sorted(CLASS_NAMES.keys())


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.

    Returns
    -------
    dict
        Accuracy, macro precision, macro recall and macro F1-score.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        )
    }

    return metrics


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix using a fixed class order.
    """

    return confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS
    )


def print_classification_report(y_true, y_pred):
    """
    Print the classification report.
    """

    target_names = [
        CLASS_NAMES[label]
        for label in CLASS_LABELS
    ]

    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=target_names,
            zero_division=0
        )
    )
