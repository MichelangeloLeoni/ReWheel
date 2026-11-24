"""Module containing functions to compute backpropagation deltas."""
import numpy as np
from . import activation_functions as af

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
        af.af_list[str(act_func) + "_DERIVATIVE"](computed_net)
        )
