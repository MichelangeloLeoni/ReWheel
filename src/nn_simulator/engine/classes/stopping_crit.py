"""Module defining the StoppingCriteria dataclass."""
from dataclasses import dataclass

@dataclass
class StoppingCriteria:
    """
    Define the stopping criteria for neural network training.

    Attributes
    ----------
    epochs : int or None, default=None
        Maximum number of training epochs. If None, this criterion
        is not considered.
    min_delta_loss : tuple[float, int] or None, default=None
        Minimum required change in training loss and the number
        of consecutive epochs over which the change is evaluated.
        If None, this criterion is not considered.
    min_gradient_norm : tuple[float, int] or None, default=None
        Minimum gradient norm and the number of consecutive epochs
        for which the gradient norm must satisfy the specified
        condition. If None, this criterion is not considered.
    min_delta_valid_loss : tuple[float, int] or None, default=None
        Minimum required change in validation loss and the number
        of consecutive epochs over which the change is evaluated.
        If None, this criterion is not considered.
    target_loss : float or None, default=None
        Target training loss. Training stops when this value is
        reached. If None, this criterion is not considered.

    Raises
    ------
    ValueError
        If no stopping criterion is specified.

    Notes
    -----
    At least one stopping criterion must be specified when
    creating an instance of this class.

    The tuple-based criteria specify a threshold and a number
    of consecutive epochs used to evaluate the corresponding
    stopping condition.
    """

    epochs: int | None = None
    min_delta_loss: tuple[float, int] | None = None
    min_gradient_norm: tuple[float, int] | None = None
    min_delta_valid_loss: tuple[float, int] | None = None
    target_loss: float | None = None

    def __post_init__(self):
        if (
            self.epochs is None
            and self.min_delta_loss is None
            and self.min_gradient_norm is None
            and self.min_delta_valid_loss is None
            and self.target_loss is None
        ):
            raise ValueError(
                "At least one stopping criterion must be specified."
            )
