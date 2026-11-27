"""Module containing functions to compute loss."""
import numpy as np
from ..error import compute_error

def compute_hmse(
        output_vectors: list[np.ndarray],
        target_vector: np.ndarray
        ) -> float:
    """
    Compute the Half Mean Squared Error over a batch of examples.

    Parameters
    ----------
    output_vectors : list of np.ndarray or np.ndarray
        Output vector of each layer in the neural network.
        If a list or tuple is provided, the last element is used
        as the final prediction. A single output array can also
        be provided directly.
    target_vector : np.ndarray
        Target vectors containing the expected output values.

    Returns
    -------
    float
        Mean Half Squared Error computed over the batch.

    Notes
    -----
    The factor 1/2 simplifies the derivative of the loss
    during backpropagation.
    """
    error_vector = compute_error(output_vectors, target_vector)

    return float(0.5 * np.mean(np.sum(error_vector**2, axis=0)))

def compute_binary_cross_entropy(
        output_vectors: list[np.ndarray],
        target_vector: np.ndarray
        ) -> float:
    """
    Compute the Binary Cross Entropy over a batch of examples.

    Parameters
    ----------
    output_vectors : list of np.ndarray or np.ndarray
        Output vector of each layer in the neural network.
        If a list or tuple is provided, the last element is used
        as the final prediction. A single output array can also
        be provided directly.
    target_vector : np.ndarray
        Target vectors containing binary labels (0 or 1).

    Returns
    -------
    float
        Mean Binary Cross Entropy computed over the batch.

    Notes
    -----
    Probabilities are clipped to avoid numerical issues when
    evaluating the logarithm at the boundaries of the interval.
    """
    prediction = (
        output_vectors[-1]
        if isinstance(output_vectors, (list, tuple))
        else output_vectors
    )

    # Clip the probabilities to avoid log(0)
    epsilon = np.finfo(float).eps # pylint: disable=no-member
    prediction = np.clip(prediction, epsilon, 1 - epsilon)

    loss = -np.mean(
        np.sum(
            target_vector * np.log(prediction)
            + (1 - target_vector) * np.log(1 - prediction),
            axis=0
        )
    )

    return float(loss)

def compute_categorical_cross_entropy(
        output_vectors: list[np.ndarray],
        target_vector: np.ndarray
        ) -> float:
    """
    Compute the Categorical Cross Entropy over a batch of examples.

    Parameters
    ----------
    output_vectors : list of np.ndarray or np.ndarray
        Output vector of each layer in the neural network.
        If a list or tuple is provided, the last element is used
        as the final prediction. A single output array can also
        be provided directly.
    target_vector : np.ndarray
        Target vectors, typically represented using one-hot
        encoding.

    Returns
    -------
    float
        Mean Categorical Cross Entropy computed over the batch.

    Notes
    -----
    Probabilities are clipped to avoid numerical issues when
    evaluating the logarithm at zero.
    """
    prediction = (
        output_vectors[-1]
        if isinstance(output_vectors, (list, tuple))
        else output_vectors
    )

    # Clip the probabilities to avoid log(0)
    epsilon = np.finfo(float).eps # pylint: disable=no-member
    prediction = np.clip(prediction, epsilon, 1.0)

    loss = -np.mean(
        np.sum(
            target_vector * np.log(prediction),
            axis=0
        )
    )

    return float(loss)
