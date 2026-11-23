"""Module containing functions to compute errors and losses."""
import numpy as np

def compute_error(
        output_vectors: list[np.ndarray],
        target_vector: np.ndarray
        ) -> np.ndarray:
    """
    Computes the residual error of a neural network prediction.

    Parameters
    ----------
    output_vectors : list of np.ndarray
        Output vector of each layer in the neural network
        (output_vector[-1] represents the final prediction).
    target_vector : np.ndarray
        Target vector for the training example.

    Returns
    -------
    error : np.ndarray
        Difference between prediction and target.
    """
    # Automatically select the output layer's output vector from the list
    if isinstance(output_vectors, (list, tuple)):
        prediction = output_vectors[-1]
    else:
        prediction = output_vectors

    return target_vector - prediction

def compute_loss(
        output_vectors: list[np.ndarray],
        target_vector: np.ndarray,
        ) -> float:
    """
    Computes the total empirical loss (HMSE) 
    over a batch of examples.

    Parameters
    ----------
    output_vectors : list of np.ndarray
        Output vector of each layer in the neural network
        (output_vector[-1] represents the final prediction).
    target_vector : np.ndarray
        Target vector for the training example.

    Returns
    -------
    loss : float
        Total loss computed as the Half Mean Squared Error (HMSE).
    """
    error_vector = compute_error(
        output_vectors=output_vectors,
        target_vector=target_vector
        )

    loss = 0.5*(np.mean(np.sum(error_vector**2, axis=0)))

    return loss
