"""Module containing functions to compute errors and losses."""
import numpy as np

def compute_error(
        output_vectors: list[np.ndarray],
        target_vector: np.ndarray
        ) -> np.ndarray:
    """
    Compute the residual error of a neural network prediction.

    The residual error is defined as the difference between the
    target vector and the predicted output.

    Parameters
    ----------
    output_vectors : list of np.ndarray or np.ndarray
        Output vector of each layer in the neural network.
        If a list or tuple is provided, the last element is used
        as the final prediction. A single output array can also
        be provided directly.
    target_vector : np.ndarray
        Target vector containing the expected output values.

    Returns
    -------
    np.ndarray
        Residual error.
    """
    prediction = (
        output_vectors[-1]
        if isinstance(output_vectors, (list, tuple))
        else output_vectors
    )

    return target_vector - prediction
