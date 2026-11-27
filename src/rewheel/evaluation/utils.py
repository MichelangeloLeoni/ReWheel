"""Module containing utilities functions for neural network evaluation."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

# NeuralNetwork is only imported during type checking and
# ignored during runtime, preventing circular imports while
# allowing type hinting
if TYPE_CHECKING:
    from ..engine.nn_sim import NeuralNetwork

def get_classification_type(
        network: "NeuralNetwork"
        ) -> str:
    """
    Determine the classification type from a neural network architecture.

    The classification type is inferred from the number of output units
    and the activation function of the output layer.

    Parameters
    ----------
    network : NeuralNetwork
        Neural network whose output configuration is inspected.

    Returns
    -------
    str
        Classification type:
        - 'B' for binary classification.
        - 'ML' for multilabel classification.
        - 'MC' for multiclass classification.

    Raises
    ------
    ValueError
        If the output configuration is not supported for classification.
    """
    output_units = network.architecture.layers[-1]
    out_act = network.architecture.act_funcs[-1]

    if output_units == 1 and out_act == 'SIGMOID':
        return 'B'
    if out_act == 'SIGMOID':
        return 'ML'
    if out_act == 'SOFTMAX':
        return 'MC'

    raise ValueError(
        "Unsupported output configuration for classification."
    )

def infer_classification_type(
        target_vector: np.ndarray
        ) -> str:
    """
    Infer the classification type from the target data.

    The classification type is determined from the target values and
    their arrangement. Targets must use binary or one-hot encoding.

    Parameters
    ----------
    target_vector : np.ndarray
        Target data. Expected to have shape (n_features, n_samples)
        or be a one-dimensional array.

    Returns
    -------
    str
        Inferred classification type:
        - 'B' for binary classification.
        - 'ML' for multilabel classification.
        - 'MC' for multiclass classification.

    Raises
    ------
    ValueError
        If the target data contains values other than 0 and 1.
    """
    if not np.all(np.isin(target_vector, [0, 1])):
        raise ValueError(
            "Cannot infer classification type from target data."
        )

    if target_vector.ndim == 1:
        if target_vector.size == 1:
            return 'B'
        if np.sum(target_vector) == 1:
            return 'MC'
        return 'ML'

    if target_vector.shape[0] == 1:
        return 'B'

    active_labels = np.sum(target_vector, axis=0)
    if np.all(active_labels == 1):
        return 'MC'

    return 'ML'

def compute_confusion_matrix(
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
