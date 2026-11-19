"""Module containing the class responsible for training the neural network."""
from typing import TYPE_CHECKING
from datetime import datetime
import numpy as np
from ..classes import TrainingLog
from ..utils import initialize_weights
from .steps import propagate, update
from .stopping import StoppingManager

# NeuralNetwork is only imported during type checking and
# ignored during runtime, preventing circular imports while
# allowing type hinting.
if TYPE_CHECKING:
    from ..nn_sim import NeuralNetwork

class Trainer:
    """
    Coordinates the training process of a neural network.

    The Trainer manages weight initialization, the training loop,
    parameter updates, and stopping criteria evaluation. It delegates
    forward and backward propagation to the appropriate functions and
    uses a StoppingManager to monitor the training progress.

    Parameters
    ----------
    neural_network : NeuralNetwork
        Neural network to be trained. The Trainer operates directly on
        its weights, architecture, and training parameters.

    Attributes
    ----------
    nn : NeuralNetwork
        Neural network being trained.
    training_log : TrainingLog
        Log containing training metrics, initialization information,
        and stopping information for the current training process.
    """
    def __init__(
            self,
            neural_network: "NeuralNetwork"
            ):
        """
        Initializes the Trainer and prepares the neural network for training.

        Initializes the network weights according to the architecture
        and training parameters, then creates a TrainingLog instance
        to record the training process.

        Parameters
        ----------
        neural_network : NeuralNetwork
            Neural network to be trained.
        """
        self.nn = neural_network

        # Initialize weights matrices
        seed = initialize_weights(
            architecture=self.nn.architecture,
            weights=self.nn.weights,
            bounds=self.nn.training_params.bounds,
            seed=self.nn.training_params.seed
        )

        # Initialize TrainingLog instance
        self.training_log = TrainingLog(
            seed=seed,
            loss=[],
            stopping_cause=[],
            validation_loss=(
                []
                if (self.nn.stopping_crit.min_delta_valid_loss is not None)
                else None
            )
        )

    def training_loop(
            self,
            input_vector: np.ndarray,
            target_vector: np.ndarray,
            valid_input_vector: np.ndarray | None = None,
            valid_target_vector: np.ndarray | None = None
            ) -> TrainingLog:
        """
        Executes the neural network training loop.

        At each epoch, the method:
        1. Computes weight gradients and training loss through propagation.
        2. Evaluates the configured stopping criteria.
        3. Stops training if one or more stopping criteria are met.
        4. Updates the network weights using the computed gradients.

        Momentum-related weight updates are handled by retaining the
        previous weight deltas between epochs.

        The training start and end timestamps are recorded in the
        associated TrainingLog.

        Parameters
        ----------
        input_vector : np.ndarray
            Training input data.
        target_vector : np.ndarray
            Target values corresponding to the training input data.
        valid_input_vector : np.ndarray or None, optional
            Validation input data used to compute validation loss when
            validation-based early stopping is enabled. Defaults to None.
        valid_target_vector : np.ndarray or None, optional
            Target values corresponding to the validation input data.
            Defaults to None.

        Returns
        -------
        TrainingLog
            Training log containing the recorded losses, validation
            losses (if enabled), stopping causes, random seed, and
            training timestamps.
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

        # Loop until a stopping condition is met
        while True:
            # Propagate trough the network
            weights_gradients, loss = propagate(
                nn=self.nn,
                input_vector=input_vector,
                target_vector=target_vector
            )

            # Compute the quantities needed for the stopping
            # creiteria, then check if anyone of them is met
            st.check_stopping(
                valid_input_vector=valid_input_vector,
                valid_target_vector=valid_target_vector,
                loss=loss,
                weights_gradients=weights_gradients
            )

            # Break the cycle if a stopping condition is met
            if self.training_log.stopping_cause:
                break

            # Update the network weights
            previous_weights_deltas = update(
                nn=self.nn,
                weights_gradients=weights_gradients,
                previous_weights_deltas=previous_weights_deltas,
                input_vector=input_vector,
                target_vector=target_vector,
                seed=self.training_log.seed
            )

        # Store training stopping time in training_log
        self.training_log.training_start_stop.append(datetime.now())

        return self.training_log
