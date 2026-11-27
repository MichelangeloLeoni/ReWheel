"""Module containing functions to compute backpropagation deltas."""
import numpy as np
from .activation_functions import af_list
from .error import compute_error
from .classes import ForwardResults

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
        forward_results: ForwardResults,
        target_vector: np.ndarray,
        act_func: str,
        loss: str
    ):
    """
    Computes the output layer delta according to the selected loss
    function and output activation function.

    Parameters
    ----------
    forward_results : ForwardResults
        Results of the forward propagation, containing the network
        predictions and the net input of each layer.
    target_vector : np.ndarray
        Target values corresponding to the input samples.
    act_func : str
        Activation function used by the output layer, represented
        by its corresponding name.
    loss : str
        Loss function used to compute the training error.
        Supported loss functions are HMSE, BCE and CCE.

    Returns
    -------
    output_delta : np.ndarray
        Delta vector of the output layer, used to initialize
        backpropagation. Its computation depends on the selected
        loss function:

        - HMSE: computes the delta by applying the derivative
          of the output activation function to the error vector.
        - BCE and CCE: directly uses the error vector as the
          output delta.
    """
    act_func = (
        act_func[-1]
        if isinstance(act_func, (list, tuple))
        else act_func
    )

    error_vector = compute_error(
        output_vectors=forward_results.output_vectors,
        target_vector=target_vector
    )

    output_delta = None
    if loss == 'HMSE':
        output_delta = compute_delta(
            propagated_error=error_vector,
            act_func=act_func,
            computed_net=forward_results.computed_nets[-1]
        )

    if loss in ('BCE', 'CCE'):
        output_delta = error_vector

    return output_delta
