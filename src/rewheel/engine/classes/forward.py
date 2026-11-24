"""Module defining the ForwardResult dataclass."""
from dataclasses import dataclass
import numpy as np

@dataclass
class ForwardResult:
    """
    Stores the intermediate results of a forward propagation.

    Attributes
    ----------
    outputs : list of np.ndarray
        Output vectors of each layer.
    nets : list of np.ndarray
        Net inputs of each layer before activation.
    """
    output_vectors: list[np.ndarray]
    computed_nets: list[np.ndarray]
