
"""Module for managing neural network training stopping criteria."""
from typing import TYPE_CHECKING
import numpy as np
from ..classes import TrainingLog
from ..utils import compute_gradient_norm
from .steps import propagate_valid

# NeuralNetwork is only imported during type checking and
# ignored during runtime, preventing circular imports while
# allowing type hinting
if TYPE_CHECKING:
    from ..nn_sim import NeuralNetwork

class StoppingManager:
    """
    Manages stopping criteria and early stopping during neural network training.

    The manager evaluates the configured stopping criteria at the end of
    each epoch, records the corresponding stopping causes, and handles
    weight restoration when early stopping is triggered.

    Parameters
    ----------
    nn : NeuralNetwork
        Neural network being trained. Provides access to its weights and
        configured stopping criteria.
    training_log : TrainingLog
        Training log used to record losses, validation losses, and
        stopping information.

    Attributes
    ----------
    nn : NeuralNetwork
        Neural network being trained.
    training_log : TrainingLog
        Training log associated with the training process.
    gradient_norms : list[float]
        Gradient norms computed at the end of each epoch.
    best_weights : list[np.ndarray] or None
        Copy of the network weights corresponding to the best validation
        loss observed so far. None until the first validation improvement.
    best_epoch : int or None
        Epoch at which the best validation loss was achieved.
    best_validation_loss : float
        Lowest validation loss observed so far.
    epochs_no_improv : int
        Number of consecutive epochs without a sufficient improvement
        in validation loss.
    """
    def __init__(
            self,
            nn: "NeuralNetwork",
            training_log: TrainingLog
            ):
        self.nn = nn
        self.training_log = training_log

        # Initialize stopping criteria states attributes
        self.gradient_norms = []
        self.best_weights: list[np.ndarray] | None = None
        self.best_epoch: int | None = None
        self.best_validation_loss: float = np.inf
        self.epochs_no_improv: int = 0

    def update_early_stopping(
            self
            ) -> None:
        """
        Updates the early stopping state using the latest validation loss.

        A new best model is recorded if the validation loss improves by
        more than the configured minimum improvement threshold. Otherwise,
        the number of consecutive epochs without sufficient improvement
        is incremented.

        When an improvement occurs, the current network weights are copied
        and the corresponding epoch is stored in the training log.
        """
        min_delta, _ = self.nn.stopping_crit.min_delta_valid_loss

        if self.training_log.metrics.validation_loss[-1] < self.best_validation_loss - min_delta:
            self.best_validation_loss = self.training_log.metrics.validation_loss[-1]
            self.epochs_no_improv = 0

            self.best_weights = [
                w.copy() for w in self.nn.weights
            ]
            self.best_epoch = len(self.training_log.metrics.validation_loss)
            self.training_log.best_epoch = self.best_epoch
        else:
            self.epochs_no_improv += 1

    def check_criteria(
            self
            ) -> list[str]:
        """
        Evaluates all configured training stopping criteria.

        The following criteria are checked:
        - Maximum number of epochs.
        - Target training loss.
        - Minimum variation in training loss.
        - Minimum gradient norm.
        - Early stopping based on validation loss.

        If early stopping is triggered, the network weights are restored
        to those corresponding to the best validation loss.

        Returns
        -------
        list[str]
            List of stopping causes triggered during the current epoch.
            An empty list indicates that no stopping criterion was met.
        """
        causes = []

        # Maximum number of epochs
        if (
            self.nn.stopping_crit.epochs is not None
            and len(self.training_log.metrics.loss) >= self.nn.stopping_crit.epochs
        ):
            causes.append("Maximum_epochs_reached")

        # Target loss
        if (
            self.nn.stopping_crit.target_loss is not None
            and self.training_log.metrics.loss[-1] <= self.nn.stopping_crit.target_loss
        ):
            causes.append("Target_loss_reached")

        # Minimum loss variation
        if self.nn.stopping_crit.min_delta_loss is not None:
            threshold, patience = self.nn.stopping_crit.min_delta_loss

            if len(self.training_log.metrics.loss) >= patience + 1:
                delta_loss = np.abs(np.diff(self.training_log.metrics.loss[-(patience + 1):]))

                if np.all(delta_loss <= threshold):
                    causes.append("Minimum_loss_variation_reached")

        # Minimum gradient norm
        if self.nn.stopping_crit.min_gradient_norm is not None:
            threshold, patience = self.nn.stopping_crit.min_gradient_norm

            if len(self.gradient_norms) >= patience:
                if np.all(np.asarray(self.gradient_norms[-patience:]) <= threshold):
                    causes.append("Minimum_gradient_norm_reached")

        # Validation based early stopping
        if self.nn.stopping_crit.min_delta_valid_loss is not None:
            _, patience = self.nn.stopping_crit.min_delta_valid_loss

            if self.epochs_no_improv >= patience:
                causes.append("Minimum_validation_loss_variation_reached (ES)")

                self.training_log.best_epoch = self.best_epoch
                self._restore_weights()

        return causes

    def _restore_weights(
            self
            ) -> None:
        """
        Restores the network weights to the best validation checkpoint.
        """
        if self.best_weights is not None:
            self.nn.weights[:] = [
                w.copy() for w in self.best_weights
            ]

    def check_stopping(
            self,
            valid_input_vector: np.ndarray,
            valid_target_vector: np.ndarray,
            loss: float,
            weights_gradients: list[np.ndarray],
            accuracy: bool = False
            ) -> None:
        """
        Updates training metrics and checks whether training should stop.

        At the end of each epoch, this method:
        1. Records the training loss.
        2. Computes and records the validation loss, if early stopping
           based on validation loss is enabled.
        3. Updates the early stopping state.
        4. Computes and records the gradient norm, if the corresponding
           stopping criterion is enabled.
        5. Evaluates all stopping criteria and updates the training log
           with the resulting stopping causes.

        Parameters
        ----------
        valid_input_vector : np.ndarray
            Input data used to evaluate the validation loss.
        valid_target_vector : np.ndarray
            Target values corresponding to the validation input data.
        loss : float
            Training loss computed during the current epoch.
        weights_gradients : list[np.ndarray]
            Weight gradient matrices computed during the current epoch.
        """
        self.training_log.metrics.loss.append(loss)

        if self.nn.stopping_crit.min_delta_valid_loss is not None:
            valid_loss, acc = propagate_valid(
                nn=self.nn,
                input_vector=valid_input_vector,
                target_vector=valid_target_vector,
                accuracy=accuracy
            )
            self.training_log.metrics.validation_loss.append(valid_loss)
            if accuracy:
                self.training_log.metrics.validation_accuracy.append(acc)
            self.update_early_stopping()

        if self.nn.stopping_crit.min_gradient_norm is not None:
            self.gradient_norms.append(compute_gradient_norm(weights_gradients))

        causes = self.check_criteria()

        self.training_log.stopping_cause = causes
