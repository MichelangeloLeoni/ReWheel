
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
        validation metrics.
    seed : int or None, default=None
        Random seed used to initialize the network weights.
    stopping_cause : list[str] or None, default=None
        Stopping criteria satisfied when training terminates.
    training_start_stop : tuple[datetime, datetime] or None, default=None
        Start and stop timestamps of the training session.
    best_epoch : int or None, default=None
        Epoch associated with the best validation performance.

    Notes
    -----
    The class stores the information required to inspect and
    reproduce a training session.

    Training and validation metrics are stored in the ``metrics``
    attribute, an instance of the ``Metrics`` dataclass.

    The number of epochs is determined by the length of the
    recorded training metrics.

    The average epoch computation time is calculated as the
    total training time divided by the number of epochs.
    """
    metrics: Metrics | None = None
    seed: int | None = None
    stopping_cause: list[str] | None = None
    training_start_stop: tuple[datetime, datetime] | None = None
    best_epoch: int | None = None

    def __repr__(self):
        epochs = None
        training_time = None
        avg_epoch_time = None
        training_start = None
        training_stop = None

        # Determine the number of epochs.
        if self.metrics is not None and self.metrics.training:
            first_metric = next(iter(self.metrics.training.values()))
            epochs = len(first_metric)

        # Compute training duration.
        if self.training_start_stop is not None:
            training_start, training_stop = self.training_start_stop

            elapsed_seconds = (
                training_stop - training_start
            ).total_seconds()

            hours, remainder = divmod(int(elapsed_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)

            training_time = (
                f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"
            )

            if epochs:
                avg_seconds = elapsed_seconds / epochs

                hours, remainder = divmod(int(avg_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)

                avg_epoch_time = (
                    f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"
                )

        stdout = [
            "TrainingLog(",
            f"    seed={self.seed},",
            f"    epochs={epochs},",
            f"    training_time={training_time},",
            f"    avg_epoch_time={avg_epoch_time},",
            f"    training_start={training_start},",
            f"    training_stop={training_stop},",
            f"    stopping_cause={self.stopping_cause},",
            f"    best_epoch={self.best_epoch},",
        ]

        if self.metrics is not None:
            for metric_type, metric_values in (
                ("training", self.metrics.training),
                ("validation", self.metrics.validation),
            ):
                if not metric_values:
                    continue

                for name, values in metric_values.items():
                    if values is None or len(values) == 0:
                        continue

                    prefix = (
                        "" if metric_type == "training"
                        else "validation_"
                    )

                    stdout.extend([
                        f"    initial_{prefix}{name.lower()}={values[0]},",
                        f"    final_{prefix}{name.lower()}={values[-1]},",
                    ])

        stdout.append(")")
        return "\n".join(stdout)
