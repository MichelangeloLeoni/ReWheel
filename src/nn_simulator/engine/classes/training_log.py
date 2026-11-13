"""Module defining the TrainingLog dataclass."""
from datetime import datetime
from dataclasses import dataclass
from .training_state import TrainingState

@dataclass
class TrainingLog:
    """
    Store the results and metadata of a neural network training session.

    Attributes
    ----------
    seed : int or None, default=None
        Random seed used to initialize the network weights.
        Can be reused to reproduce the same initialization.
    loss : list[float] or None, default=None
        Loss values recorded during training. Each element
        corresponds to the loss evaluated at a training epoch.
    stopping_cause : list[str] or None, default=None
        Stopping criteria satisfied when training terminates.
        Multiple criteria may be satisfied simultaneously.
    training_start_stop : tuple[datetime, datetime] or None, default=None
        Start and stop timestamps of the training session.
    validation_loss : list[float] or None, default=None
        Validation loss values recorded during training.
        Each element corresponds to the validation loss evaluated
        at a training epoch.
    best_epoch : int or None, default=None
        Epoch associated with the lowest validation loss.
        Used to identify the best model according to its
        generalization performance.
    training_state : TrainingState or None, default=None
        Store an instance of TrainingState providing
        access to training state variables for debugging purposes.

    Notes
    -----
    The class stores the information required to inspect and
    reproduce a training session.

    Several useful quantities can be derived from its attributes:

    - Number of training epochs: ``len(loss)``.
    - Initial training loss: ``loss[0]``.
    - Final training loss: ``loss[-1]``.
    - Initial validation loss: ``validation_loss[0]``.
    - Final validation loss: ``validation_loss[-1]``.
    - Best validation loss: ``min(validation_loss)``.
    - Elapsed training time: difference between the stop and
      start timestamps in ``training_start_stop``.

    The ``stopping_cause`` attribute can contain multiple entries
    when different stopping criteria are satisfied simultaneously.

    If a validation set is provided, ``best_epoch`` identifies
    the epoch at which the validation loss reaches its minimum.
    This information can be used for model selection and early
    stopping.
    """

    seed: int | None = None
    loss: list[float] | None = None
    stopping_cause: list[str] | None = None
    training_start_stop: tuple[datetime, datetime] | None = None
    validation_loss: list[float] | None = None
    best_epoch: int | None = None
    training_state: TrainingState | None = None

    def __repr__(self):
        """Return a formatted summary of the training session."""
        elapsed_time = None
        training_start = None
        training_stop = None
        elapsed_epoch = len(self.loss) if self.loss is not None else None

        if self.training_start_stop is not None:
            training_start, training_stop = self.training_start_stop
            elapsed_time = (
                training_stop - training_start
            ).total_seconds()

        stdout = [
            "TrainingLog(",
            f"    seed={self.seed},",
            f"    elapsed_epoch={elapsed_epoch},",
            f"    elapsed_time={elapsed_time:.3f}s,"
            if elapsed_time is not None
            else "    elapsed_time=None,",
            f"    training_start={training_start},",
            f"    training_stop={training_stop},",
            f"    stopping_cause={self.stopping_cause},",
        ]

        if self.loss is not None:
            stdout.extend([
                f"    initial_loss={self.loss[0]},",
                f"    final_loss={self.loss[-1]},",
            ])

        if self.validation_loss is not None:
            stdout.extend([
                f"    initial_validation_loss={self.validation_loss[0]},",
                f"    final_validation_loss={self.validation_loss[-1]},",
                f"    best_epoch={self.best_epoch},",
            ])

        stdout.append(")")
        return "\n".join(stdout)
