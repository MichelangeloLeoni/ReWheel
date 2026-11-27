"""Module containing the class responsible for training the neural network."""
from typing import TYPE_CHECKING
from collections import deque
from datetime import datetime
import numpy as np
from rich.live import Live
from rich.console import Group
from rich.text import Text
from ..classes import TrainingLog
from ..utils import initialize_weights
from .steps import (
    propagate,
    propagate_valid,
    update_weights,
    update_mini_batches
)
from .stopping import StoppingManager
from ..classes import Metrics

# NeuralNetwork is only imported during type checking and
# ignored during runtime, preventing circular imports while
# allowing type hinting
if TYPE_CHECKING:
    from ..nn_sim import NeuralNetwork

class Trainer:
    """
    Coordinates the training process of a neural network.

    The Trainer initializes the network weights, manages the training
    loop, performs weight updates, and monitors the configured stopping
    criteria.

    Parameters
    ----------
    neural_network : NeuralNetwork
        Neural network to be trained. The Trainer operates directly on
        its weights, architecture, training parameters, and stopping
        criteria.

    Attributes
    ----------
    nn : NeuralNetwork
        Neural network being trained.
    training_log : TrainingLog
        Log containing training metrics, validation metrics (if enabled),
        initialization information, and stopping information.
    _progress_logs : deque
        Buffer retaining the ten most recent training progress messages.
    """

    def __init__(
            self,
            neural_network: "NeuralNetwork",
            metrics: tuple[str, ...]
            ):
        """
        Initializes the Trainer and prepares the neural network for training.

        Initializes the network weights according to its architecture
        and training parameters.

        Parameters
        ----------
        neural_network : NeuralNetwork
            Neural network to be trained.
        """
        self.nn = neural_network
        self.metrics = metrics

        # Initialize weights matrices
        seed = initialize_weights(
            architecture=self.nn.architecture,
            weights=self.nn.weights,
            bounds=self.nn.training_params.bounds,
            seed=self.nn.training_params.seed
        )

        # Initialize TrainingLog instance
        self.training_log = TrainingLog(
            metrics=Metrics(
                training={},
                validation=(
                    {}
                    if self.nn.stopping_crit.min_delta_valid_loss is not None
                    else None
                )
            ),
            seed=seed,
            stopping_cause=[]
        )

        # Initialize progress logs buffer
        self._progress_logs = deque(maxlen=10)

    def _progress(self, live: Live) -> None:
        """Display the last 10 training progress updates."""

        metrics = self.training_log.metrics

        epoch = len(
            next(iter(metrics.training.values()))
        )

        message = f"Epoch {epoch}"

        for name, values in metrics.training.items():
            message += f", {name}: {values[-1]:.5g}"

        if metrics.validation is not None:
            for name, values in metrics.validation.items():
                message += f", validation {name}: {values[-1]:.5g}"

        self._progress_logs.append(Text(message))
        live.update(Group(*self._progress_logs), refresh=True)

    def training_loop(
            self,
            input_vector: np.ndarray,
            target_vector: np.ndarray,
            valid_input_vector: np.ndarray | None = None,
            valid_target_vector: np.ndarray | None = None
            ) -> TrainingLog:
        """
        Executes the neural network training loop.

        At each training iteration, the method:
        1. Computes weight gradients and training loss through propagation.
        2. Evaluates the configured stopping criteria using a
           StoppingManager.
        3. Stops training if one or more stopping criteria are met.
        4. Updates the network weights using full-batch or mini-batch
           gradient descent.

        For mini-batch training, the training samples are shuffled and
        divided into batches before the weight updates are performed.

        Momentum-related updates are handled by retaining the weight
        deltas from the previous update step.

        The training start and end timestamps are recorded in the
        associated TrainingLog.

        Parameters
        ----------
        input_vector : np.ndarray
            Training input samples, arranged in columns.
        target_vector : np.ndarray
            Target values corresponding to the training samples,
            arranged in columns.
        valid_input_vector : np.ndarray or None, optional
            Validation input samples, arranged in columns. Used to
            evaluate validation loss when validation-based stopping
            criteria are enabled. Defaults to None.
        valid_target_vector : np.ndarray or None, optional
            Target values corresponding to the validation samples,
            arranged in columns. Defaults to None.

        Returns
        -------
        TrainingLog
            Training log containing the recorded training losses,
            validation losses (if enabled), stopping causes, random
            seed, and training timestamps.

        Notes
        -----
        Training progress is displayed in real time in the console,
        including training and validation losses when available.

        The training log is printed automatically after training
        is completed.

        The training loop terminates when at least one configured
        stopping criterion is met.
        """
        # Initialize previous weights deltas to zero
        previous_weights_deltas = [
            np.zeros_like(w) for w in self.nn.weights
        ]

        # Initialize an instance of StoppingManager
        st = StoppingManager(
            nn=self.nn,
            training_log=self.training_log
        )

        # Store training starting time in training_log
        self.training_log.training_start_stop = [datetime.now()]

        with Live(
            Group(),
            refresh_per_second=10,
            transient=False
        ) as live:

            # Loop until a stopping condition is met
            while True:
                # Propagate through the network
                weights_gradients, computed_metrics = propagate(
                    nn=self.nn,
                    input_vector=input_vector,
                    target_vector=target_vector,
                    metrics=self.metrics
                )
                self.training_log.metrics.record(
                    computed_metrics=computed_metrics
                )

                if self.nn.stopping_crit.min_delta_valid_loss is not None:
                    val_computed_metrics = propagate_valid(
                        nn=self.nn,
                        input_vector=valid_input_vector,
                        target_vector=valid_target_vector,
                        metrics=self.metrics
                    )
                    self.training_log.metrics.record(
                        computed_metrics=val_computed_metrics,
                        validation=True
                    )

                # Evaluate stopping criteria
                st.check_stopping(
                    weights_gradients=weights_gradients
                )

                self._progress(live)

                # Stop training if a stopping condition is met
                if self.training_log.stopping_cause:
                    break

                # Full-batch training
                if self.nn.training_params.batch_size is None:
                    previous_weights_deltas = update_weights(
                        weights=self.nn.weights,
                        weights_gradients=weights_gradients,
                        previous_weights_deltas=previous_weights_deltas,
                        update_params=self.nn.update_params
                    )

                # Mini-batch training
                else:
                    previous_weights_deltas = update_mini_batches(
                        nn=self.nn,
                        previous_weights_deltas=previous_weights_deltas,
                        input_vector=input_vector,
                        target_vector=target_vector,
                        seed=self.nn.training_params.seed
                    )

        # Store training stopping time in training_log
        self.training_log.training_start_stop.append(datetime.now())

        self.log()

        return self.training_log

    def log(
            self
            ) -> None:
        """
        Prints the training log to the console.
        """
        if self.training_log is not None:
            print(self.training_log)
