"""Module containing the functions that compute classification metrics."""
import numpy as np

def compute_accuracy(
        predicted_classes: np.ndarray,
        true_classes: np.ndarray,
        multilabel: bool = False
        ) -> float:
    """Compute classification accuracy."""

    if multilabel:
        return float(np.mean(
            np.all(predicted_classes == true_classes, axis=0)
        ))

    return float(np.mean(
        predicted_classes == true_classes
    ))

def generate_confusion_matrix(
        true_classes: np.ndarray,
        predicted_classes: np.ndarray,
        n_classes: int
        ) -> np.ndarray:
    """
    Compute the confusion matrix from true and predicted class labels.

    Rows represent true classes, while columns represent predicted
    classes. Each element indicates the number of samples assigned
    to the corresponding true-predicted class pair.

    Parameters
    ----------
    true_classes : np.ndarray
        Array containing the true class indices.
    predicted_classes : np.ndarray
        Array containing the predicted class indices.
    n_classes : int
        Number of classes. Determines the dimensions of the
        resulting confusion matrix.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes), where
        element [i, j] represents the number of samples belonging
        to class i and predicted as class j.
    """
    true_classes = np.asarray(true_classes).ravel()
    predicted_classes = np.asarray(predicted_classes).ravel()

    confusion_matrix = np.zeros(
        (n_classes, n_classes),
        dtype=int
    )

    for true_class, predicted_class in zip(
            true_classes, predicted_classes
            ):
        confusion_matrix[true_class, predicted_class] += 1

    return confusion_matrix

def compute_confusion_matrix(
        predicted_classes: np.ndarray,
        true_classes: np.ndarray,
        n_classes: int,
        multilabel: bool = False
        ) -> np.ndarray:
    """Compute the confusion matrix."""

    if multilabel:
        return np.stack([
            generate_confusion_matrix(
                true_classes[i],
                predicted_classes[i],
                2
            )
            for i in range(n_classes)
        ])

    return generate_confusion_matrix(
        true_classes,
        predicted_classes,
        n_classes
    )
