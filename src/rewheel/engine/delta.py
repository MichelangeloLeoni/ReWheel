"""Module containing functions to compute backpropagation deltas."""
import numpy as np
from .activation_functions import af_list
from .loss import compute_error

def compute_delta(
        propagated_error: np.ndarray,
        act_func: str,
        computed_net: np.ndarray
        ) -> np.ndarray:
    """
    Computes the delta vector of a layer, needed to compute the
    weights gradient when back-propagating.

    Parameters
    ----------
    propagated_error : np.ndarray
        Array containing the propagated delta coming from the previous layer.
        For the output layer is simply the residual error vector.
    act_func : str
        Activation function of the layer, represented by a
        string. Needed to compute it's derivative.
    computed_net : np.ndarray
        Net vector of the layer, previously computed during
        forwarding.

    Returns
    -------
    np.ndarray
        Computed delta vector of the layer.
    """
    # Here * performs element-wise matrix product
    return propagated_error * (
        af_list[str(act_func) + "_DERIVATIVE"](computed_net)
        )

def compute_output_delta(
        output_vectors: list[np.ndarray],
        target_vector: np.ndarray,
        computed_nets: list[np.ndarray],
        act_func: str,
        loss: str
    ):
    error_vector = compute_error(
        output_vectors=output_vectors,
        target_vector=target_vector
    )

    output_delta = None
    if loss == 'HMSE':
        output_delta = compute_delta(
            propagated_error=error_vector,
            act_func=act_func,
            computed_net=computed_nets[-1]
        )

    if loss in ('BCE', 'CCE'):
        output_delta = error_vector

    return output_delta
