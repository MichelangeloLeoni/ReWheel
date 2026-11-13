"""Module defining the TrainingState dataclass."""
from dataclasses import dataclass
import numpy as np

@dataclass
class TrainingState:
    """
    Store temporary quantities used during neural network training.

    Attributes
    ----------
    error : float or None, default=None
        Error value computed during the training procedure.
    nets : list of numpy.ndarray or None, default=None
        Net input values computed for each network layer during
        forward propagation.
    outputs : list of numpy.ndarray or None, default=None
        Output values computed for each network layer during
        forward propagation.
    weights_gradients : list of numpy.ndarray or None, default=None
        Gradients of the loss function with respect to the network
        weights, computed during backpropagation.
    previous_weights_deltas : list of numpy.ndarray or None, default=None
        Weight update increments computed during the previous
        training iteration. Used to implement momentum.
    delta_loss : list of float or None, default=None
        Absolute or signed variations in training loss between
        consecutive epochs, according to the implementation.
    gradient_norms : list of float or None, default=None
        Norms of the weight gradients recorded during training.

    Notes
    -----
    This class collects intermediate quantities required by
    the training algorithm, including forward propagation,
    backpropagation, weight updates, and convergence monitoring.

    It is primarily intended for internal use by the Trainer
    class and can also be accessed through the TrainingLog
    for debugging and inspection purposes.
    """
    error: float | None = None
    nets: list[np.ndarray] | None = None
    outputs: list[np.ndarray] | None = None
    weights_gradients: list[np.ndarray] | None = None
    previous_weights_deltas: list[np.ndarray] | None = None
    delta_loss: list[float] | None = None
    gradient_norms: list[float] | None = None
