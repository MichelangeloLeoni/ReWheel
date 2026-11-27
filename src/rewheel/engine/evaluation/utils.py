"""Module containing utilities functions for neural network evaluation."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .wrappers import (
    metric_compatibility,
    loss_functions,
    classification_metrics
)

# NeuralNetwork is only imported during type checking and
# ignored during runtime, preventing circular imports while
# allowing type hinting
if TYPE_CHECKING:
    from ...engine.nn_sim import NeuralNetwork

def get_task_type(network: "NeuralNetwork") -> str:
    output_units = network.architecture.layers[-1]
    out_act = network.architecture.act_funcs[-1]

    if out_act == 'SOFTMAX':
        return 'MC'

    if out_act == 'SIGMOID':
        return 'B' if output_units == 1 else 'ML'

    if out_act in ('LINEAR', 'RELU'):
        return 'R'

    raise ValueError(
        "Unsupported output configuration."
    )

def transform_classes(
        output_vector: np.ndarray,
        target_vector: np.ndarray,
        threshold: float,
        class_type: str
        ) -> tuple[np.ndarray, np.ndarray]:

    if class_type == 'MC':
        predicted_classes = np.argmax(output_vector, axis=0)
        true_classes = np.argmax(target_vector, axis=0)

    elif class_type == 'B':
        predicted_classes = (
            output_vector >= threshold
        ).astype(int).ravel()
        true_classes = target_vector.astype(int).ravel()

    elif class_type == 'ML':
        predicted_classes = (
            output_vector >= threshold
        ).astype(int)
        true_classes = target_vector.astype(int)

    else:
        raise ValueError(
            f"Unsupported classification type: {class_type}"
        )

    return predicted_classes, true_classes

def check_metrics(
        metrics: tuple[str, ...],
        network: "NeuralNetwork"
        ) -> tuple[str, ...]:

    # Normalize metrics
    if isinstance(metrics, str):
        metrics = (metrics,)

    metrics = tuple(dict.fromkeys(
        metric.upper() for metric in metrics
    ))

    invalid_metrics = [
        metric for metric in metrics
        if metric not in loss_functions and metric not in classification_metrics
    ]

    if invalid_metrics:
        raise ValueError(
            f"Unsupported metrics: {invalid_metrics}"
        )

    class_type = get_task_type(
        network=network
    )

    # Check metrics compatibility
    incompatible_metrics = [
        metric for metric in metrics
        if class_type not in metric_compatibility[metric]
    ]

    if incompatible_metrics:
        raise ValueError(
            f"Metrics {incompatible_metrics} are not "
            f"compatible with classification type '{class_type}'."
        )
