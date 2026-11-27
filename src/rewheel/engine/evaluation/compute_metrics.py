"""Module containing functions used to evaluate neural network predictions."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .utils import (
    get_task_type,
    transform_classes
)
from .wrappers import loss_functions, classification_metrics
from ..classes import EvaluationParams

# NeuralNetwork is only imported during type checking and
# ignored during runtime, preventing circular imports while
# allowing type hinting
if TYPE_CHECKING:
    from ..nn_sim import NeuralNetwork

def compute_metrics(
        output_vector: np.ndarray,
        target_vector: np.ndarray,
        metrics: tuple[str, ...],
        threshold: float,
        network: "NeuralNetwork"
        ) -> dict:

    output_vector = (
        output_vector[-1]
        if isinstance(output_vector, (list, tuple))
        else output_vector
    )

    class_type = get_task_type(network)

    # supposed valid metrics value
    results = {}

    losses = list(
        metric for metric in metrics
        if metric in loss_functions
    )
    if losses:
        params = EvaluationParams(
            output_vector=output_vector,
            target_vector=target_vector,
        )
        for loss in losses:
            results[loss] = loss_functions[loss](params)

    class_metrics = list(
        metric for metric in metrics
        if metric in classification_metrics
    )
    if class_metrics:
        predicted_classes, true_classes = transform_classes(
            output_vector=output_vector,
            target_vector=target_vector,
            threshold=threshold,
            class_type=class_type
        )
        n_classes = (
            2 if class_type == 'B'
            else output_vector.shape[0]
        )
        params = EvaluationParams(
            predicted_classes=predicted_classes,
            true_classes=true_classes,
            multilabel=class_type == 'ML',
            n_classes=n_classes
        )
        for metric in class_metrics:
            results[metric] = classification_metrics[metric](params)

    return results
