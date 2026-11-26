"""Module defining the TrainingLog dataclass."""
from datetime import datetime
from dataclasses import dataclass
from .metrics import Metrics

@dataclass
class TrainingLog:
    """
    Store the results and metadata of a neural network training session.

    Attributes
    ----------
    metrics : Metrics or None, default=None
        Metrics recorded during training, including training and
        validation loss and accuracy.
    seed : int or None, default=None
        Random seed used to initialize the network weights.
        Can be reused to reproduce the same initialization.
    stopping_cause : list[str] or None, default=None
        Stopping criteria satisfied when training terminates.
        Multiple criteria may be satisfied simultaneously.
    training_start_stop : tuple[datetime, datetime] or None, default=None
        Start and stop timestamps of the training session.
    best_epoch : int or None, default=None
        Epoch associated with the lowest validation loss.
        Used to identify the best model according to its
        generalization performance.

    Notes
    -----
    The class stores the information required to inspect and
    reproduce a training session.

    Training and validation metrics are stored in the ``metrics``
    attribute, which is an instance of the ``Metrics`` dataclass.
    Available metrics include loss and accuracy.

    Several useful quantities can be derived from its attributes:

    - Number of training epochs: ``len(metrics.loss)``.
    - Initial training loss: ``metrics.loss[0]``.
    - Final training loss: ``metrics.loss[-1]``.
    - Initial training accuracy: ``metrics.accuracy[0]``.
    - Final training accuracy: ``metrics.accuracy[-1]``.
    - Initial validation loss: ``metrics.validation_loss[0]``.
    - Final validation loss: ``metrics.validation_loss[-1]``.
    - Initial validation accuracy: ``metrics.validation_accuracy[0]``.
    - Final validation accuracy: ``metrics.validation_accuracy[-1]``.
    - Best validation loss: ``min(metrics.validation_loss)``.
    - Elapsed training time: difference between the stop and
      start timestamps in ``training_start_stop``.

    Metric attributes may be ``None`` when the corresponding
    metric is not recorded.

    If a validation set is provided, ``best_epoch`` identifies
    the epoch at which the validation loss reaches its minimum.
    This information can be used for model selection and early
    stopping.
    """
    metrics: Metrics | None = None
    seed: int | None = None
    stopping_cause: list[str] | None = None
    training_start_stop: tuple[datetime, datetime] | None = None
    best_epoch: int | None = None


    def __repr__(self):
        epochs = None
        training_time = None
        training_start = None
        training_stop = None

        if self.metrics is not None and self.metrics.loss is not None:
            epochs = len(self.metrics.loss)

        if self.training_start_stop is not None:
            training_start, training_stop = self.training_start_stop
            elapsed_seconds = (training_stop - training_start).total_seconds()

            hours, remainder = divmod(int(elapsed_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)

            training_time = f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"

        stdout = [
            "TrainingLog(",
            f"    seed={self.seed},",
            f"    epochs={epochs},",
            f"    training_time={training_time},",
            f"    training_start={training_start},",
            f"    training_stop={training_stop},",
            f"    stopping_cause={self.stopping_cause},",
        ]

        if self.metrics is not None:
            metric_names = (
                ("loss", "initial_loss", "final_loss"),
                ("accuracy", "initial_accuracy", "final_accuracy"),
                ("validation_loss", "initial_validation_loss", "final_validation_loss"),
                (
                    "validation_accuracy",
                    "initial_validation_accuracy",
                    "final_validation_accuracy",
                ),
            )

            for attribute, initial_name, final_name in metric_names:
                values = getattr(self.metrics, attribute)

                if values is not None and len(values) > 0:
                    stdout.extend([
                        f"    {initial_name}={values[0]},",
                        f"    {final_name}={values[-1]},",
                    ])

            if self.metrics.validation_loss is not None:
                stdout.append(f"    best_epoch={self.best_epoch},")

        stdout.append(")")
        return "\n".join(stdout)
