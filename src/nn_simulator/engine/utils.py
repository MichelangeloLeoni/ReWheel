"""
Module containing support utilities for the functions needed
in the neural network simulator.
"""
import numpy as np
from . import activation_functions as af
from .classes import Architecture

def check_input(
        input_vector: list | np.ndarray,
        input_vector_size: int
        ) -> None:
    """
    Checks if the input vector type is np.ndarray, if not
    performs a casting, then checks if the number of features
    in the input vector matches the number of units in the input layer.

    Parameters
    ----------
    input_vector : np.ndarray
        Input vector to be forwarded into the neural network.
    input_vector_size : int
        Number of features in the input vector.

    Raises
    ------
        RuntimeError
            If the model has not been trained or loaded from memory.
        ValueError
            If the number of features in the input vector doesn't
            match the number of units in the input layer.
    """
    if not isinstance(input_vector, np.ndarray):
        input_vector = np.array(input_vector)

    if input_vector.shape[0] != input_vector_size:
        raise ValueError(
            "Input vector size does not match the network's input layer number of units."
            )

def add_bias(
        array: np.ndarray,
        is_single_example: bool
        ) -> np.ndarray:
    """
    Inserts the bias component in a np.ndarray.

    Parameters
    ----------
    array : np.ndarray
        Array to be expanded.
    is_single_example : bool
        True if a single example is being fed into the neural network,
        false otherwise.

    Returns
    -------
    np.ndarray
        Expanded array.
    """
    return (np.append(array, 1)
            if is_single_example
            else np.concatenate(
                (array, np.ones((1, array.shape[1]))),
                axis=0
                )
            )

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

def compute_gradient(
        delta: np.ndarray,
        prev_layer_output: np.ndarray,
        is_single_example: bool
    ) -> np.ndarray:
    """
    Computes the weights gradient of a layer when back-propagating.

    Parameters
    ----------
    delta : np.ndarray
        Delta vector of the layer.
    prev_layer_output : np.ndarray
        Output vector of the previous layer.
    is_single_example : bool
        True if a single example is being fed into the neural network,
        false otherwise.
    
    Returns
    -------
    np.ndarray
        Computed weights gradient of the layer.
    """
    return  (np.outer(delta, prev_layer_output)
                if is_single_example
                else delta @ prev_layer_output.T / prev_layer_output.shape[1]
    )

def initialize_weights(
        architecture: Architecture,
        weights: list[np.ndarray],
        bounds: tuple[float, float] = (-1, 1),
        seed: int = None
        ) -> int:
    """
    Initializes weights matrices with random floats
    following a uniform distribution. A seed can be
    passed as a parameter to reproduce certain
    weights starting values. If a seed is not provided
    it will be generated. The seed that was used will
    be returned.

    Parameters
    ----------
    architecture : Architecture
        Neural network architecture.
    weights : list of np.ndarray
        List of weights matrices of the neural network.
    bounds : tuple[float, float], default=(-1, 1)
        Lower and upper bounds of the uniform distribution used
        to initialize the network weights.
    seed : int or None, default=None
        Seed used to initialize the random number generator.
        If None, a random seed is generated automatically.
        The generated seed can be saved and reused to reproduce
        the same initial weights, provided that the network
        architecture and bounds remain unchanged.
    
    Returns
    -------
    seed : int
        The seed that was used to initialize the weights.
    """
    if seed is None:
        seed = np.random.SeedSequence().generate_state(
            1, dtype=np.uint32
        )[0].item()

    rng = np.random.default_rng(seed)

    for i, layer in enumerate(architecture.layers):
        if i == 0:
            weights[i] = rng.uniform(
                low=bounds[0],
                high=bounds[1],
                size=(layer, architecture.input_vector_size+1)
                )
        else:
            weights[i] = rng.uniform(
                low=bounds[0],
                high=bounds[1],
                size=(layer, architecture.layers[i-1]+1)
                )

    return seed

def compute_gradient_norm(
        weights_gradients: list[np.ndarray]
        ) -> float:
    """
    Compute the global L2 norm of the weights gradients.

    Parameters
    ----------
    weights_gradients : list of np.ndarray
        List containing the negative weights gradient matrix of each layer.

    Returns
    -------
    float
        Global L2 norm of all weights gradients.
    """
    return np.sqrt(
        sum(np.sum(grad ** 2) for grad in weights_gradients)
    )
