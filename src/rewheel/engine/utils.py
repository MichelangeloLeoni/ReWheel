"""
Module containing support utilities for the functions needed
in the neural network simulator.
"""
from functools import wraps
import numpy as np
from .classes import Architecture

def check_input(func: callable) -> callable:
    """
    Decorator that validates and converts the input vector before
    passing it to a neural network method.

    The input vector is converted to a np.ndarray if necessary.
    Its first dimension is then checked against the number of
    input units specified in the network architecture.

    Parameters
    ----------
    func : callable
        Instance method of a NeuralNetwork object to be decorated.
        The method must accept the input vector as its first
        argument after self.

    Returns
    -------
    callable
        Wrapped method that validates the input vector before
        executing the original function.

    Raises
    ------
    ValueError
        If the number of input features does not match the
        network's input layer size.
    """
    @wraps(func) # Needed to keep the original function metadata
    def wrapper(self, input_vector, *args, **kwargs):
        # Check if the input is a NumPy array,
        # if not attempt a conversion
        if not isinstance(input_vector, np.ndarray):
            input_vector = np.array(input_vector)

        # Check if input vector number of features matches input layer
        # number of units
        if input_vector.shape[0] != self.architecture.input_vector_size:
            raise ValueError(
                "Input vector size does not match the network's "
                "input layer number of units."
            )

        # Execute the decorated method
        return func(self, input_vector, *args, **kwargs)

    return wrapper

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
