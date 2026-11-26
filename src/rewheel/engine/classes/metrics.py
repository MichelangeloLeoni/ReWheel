"""Module defining the Metrics dataclass."""
from dataclasses import dataclass

@dataclass
class Metrics:
    """Container for storing training and validation metrics.

    Attributes
    ----------
    loss : list of float or None, default=None
        List containing the loss values recorded during training.
    accuracy : list of float or None, default=None
        List containing the accuracy values recorded during training.
    validation_loss : list of float or None, default=None
        List containing the loss values recorded on the validation set.
    validation_accuracy : list of float or None, default=None
        List containing the accuracy values recorded on the validation set.
    """
    loss: list[float] | None = None
    accuracy: list[float] | None = None
    validation_loss: list[float] | None = None
    validation_accuracy: list[float] | None = None
