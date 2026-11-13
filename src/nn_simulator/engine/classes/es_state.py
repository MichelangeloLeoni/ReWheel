"""Module defining the EsState dataclass."""
from dataclasses import dataclass
import numpy as np

@dataclass
class EsState:
    """
    Store the state of the early stopping procedure.

    Attributes
    ----------
    best_weights : list of np.ndarray or None, default=None
        Weights of the network corresponding to the lowest validation
        loss achieved so far.
    best_epoch : int or None, default=None
        Epoch at which the lowest validation loss was achieved.
    best_validation_loss : float, default=np.inf
        Lowest validation loss achieved so far.
    epochs_no_improv : int, default=0
        Number of consecutive epochs without an improvement in
        validation loss.
    """
    best_weights: list[np.ndarray] | None = None
    best_epoch: int | None = None
    best_validation_loss: float = np.inf
    epochs_no_improv: int = 0
