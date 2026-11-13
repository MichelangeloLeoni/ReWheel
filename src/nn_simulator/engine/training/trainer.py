"""Module containing the function that trains the neural network."""
from datetime import datetime
import numpy as np
from ..classes import (
    UpdateParameters,
    TrainingParameters,
    StoppingCriteria,
    Architecture,
    TrainingLog,
    TrainingState,
    EsState
    )
from ..loss import compute_error, compute_loss
from ..propagation import forward_propagation, backward_propagation, update_weights
from ..utils import initialize_weights, compute_gradient_norm

class Trainer:
    """
    Handle the training process of a neural network.

    This class manages weight initialization, training iterations,
    weight updates and stopping criteria. It also records the training
    history and generates a log containing the results.

    Parameters
    ----------
    weights : list of np.ndarray
        List of weights matrices of the neural network.
    architecture : Architecture
        Architecture of the neural network.
    update_params : UpdateParameters
        Parameters controlling weight updates, including learning
        rate, momentum and regularization.
    training_params : TrainingParameters
        Parameters controlling the training process and its
        stopping criteria.

    Attributes
    ----------
    weights : list of np.ndarray
        List of weights matrices of the neural network.
    architecture : Architecture
        Architecture of the neural network.
    update_params : UpdateParameters
        Parameters controlling weight updates.
    training_params : TrainingParameters
        Parameters controlling the training process.
    previous_weights_deltas : listo of np.ndarray
        List containing the weights deltas of the previous weights update
        iteration. Needed to apply momentum to the weight update.
    seed : int
        Random seed used to initialize the weights.
    """

    def __init__(
            self,
            architecture: Architecture,
            update_params: UpdateParameters,
            training_params: TrainingParameters,
            stopping_crit: StoppingCriteria,
            weights: list[np.ndarray]
            ):
        self.weights = weights
        self.architecture = architecture
        self.update_params = update_params
        self.training_params = training_params
        self.stopping_crit = stopping_crit
        # Initialize weights matrices
        seed = initialize_weights(
            architecture=architecture,
            weights=weights,
            bounds=training_params.bounds,
            seed=training_params.seed
        )
        # Initialize TrainingState instance
        self.training_state = TrainingState(
            delta_loss=[],
            gradient_norms=[],
            previous_weights_deltas = [
                np.zeros_like(w) for w in self.weights
            ]
        )
        # Initialize TrainingLog instance
        self.training_log = TrainingLog(
            seed=seed,
            loss=[],
            stopping_cause=[],
            validation_loss=(
                []
                if (self.stopping_crit.min_delta_valid_loss is not None)
                else None
            )
        )
        if self.stopping_crit.min_delta_valid_loss is not None:
            self.es_state = EsState()

    def _update_early_stopping(self, validation_loss: float) -> None:
        """Update the early stopping state using the validation loss."""

        min_delta, _ = self.stopping_crit.min_delta_valid_loss

        if validation_loss < self.es_state.best_validation_loss - min_delta:
            self.es_state.best_validation_loss = validation_loss
            self.es_state.epochs_no_improv = 0

            self.es_state.best_weights = [
                w.copy() for w in self.weights
            ]
            self.es_state.best_epoch = len(self.training_log.validation_loss)
        else:
            self.es_state.epochs_no_improv += 1

    def _check_stopping_criteria(self) -> list[str]:
        """
        Check whether any of the training stopping criteria is satisfied.

        The method checks the maximum number of epochs, target loss,
        minimum loss variation and minimum gradient norm.

        Parameters
        ----------
        loss : list of float
            Loss values recorded during training.
        gradient_norms : list of float
            Gradient norms recorded during training.

        Returns
        -------
        list of str
            Names of all stopping criteria currently satisfied.
            An empty list is returned if no criterion is satisfied.
        """
        causes = []

        # Maximum number of epochs
        if (
            self.stopping_crit.epochs is not None
            and len(self.training_log.loss) >= self.stopping_crit.epochs
        ):
            causes.append("Maximum_epochs_reached")

        # Target loss
        if (
            self.stopping_crit.target_loss is not None
            and self.training_log.loss[-1] <= self.stopping_crit.target_loss
        ):
            causes.append("Target_loss_reached")

        # Minimum loss variation
        if self.stopping_crit.min_delta_loss is not None:
            threshold, patience = self.stopping_crit.min_delta_loss

            if len(self.training_log.loss) >= patience + 1:
                delta_loss = np.abs(np.diff(self.training_log.loss[-(patience + 1):]))

                if np.all(delta_loss <= threshold):
                    causes.append("Minimum_loss_variation_reached")

        # Minimum gradient norm
        if self.stopping_crit.min_gradient_norm is not None:
            threshold, patience = self.stopping_crit.min_gradient_norm

            if len(self.training_state.gradient_norms) >= patience:
                if np.all(np.array(self.training_state.gradient_norms[-patience:]) <= threshold):
                    causes.append("Minimum_gradient_norm_reached")

        if self.stopping_crit.min_delta_valid_loss is not None:
            _, patience = self.stopping_crit.min_delta_valid_loss

            if self.es_state.epochs_no_improv >= patience:
                causes.append("Minimum_validation_loss_variation_reached (ES)")

        self.training_log.stopping_cause = causes

    def loss_wrapper(
            self,
            input_vector: np.ndarray,
            target_vector: np.ndarray
            ) -> float:
        """
        Propagate and compute the loss for a batch of examples.

        Parameters
        ----------
        input_vector : np.ndarray
            Input vector to be forwarded into the neural network.
        target_vector : np.ndarray
            Target vector for the training example.

        Returns
        -------
        loss : float
            Loss computed for the current batch.
        """
        self.training_state.outputs, self.training_state.nets = forward_propagation(
            weights=self.weights,
            act_funcs=self.architecture.act_funcs,
            input_vector=input_vector
        )

        self.training_state.error = compute_error(
            output_vectors=self.training_state.outputs,
            target_vector=target_vector
        )

        loss = compute_loss(
            output_vectors=self.training_state.outputs,
            target_vector=target_vector,
            error_vector=self.training_state.error
        )

        return loss

    def _backprop_wrapper(
            self
            ) -> None:
        self.training_state.weights_gradients = backward_propagation(
            weights=self.weights,
            act_funcs=self.architecture.act_funcs,
            output_vectors=self.training_state.outputs,
            computed_nets=self.training_state.nets,
            error_vector=self.training_state.error
        )

    def _update_wrapper(
            self
            ) -> None:
        """
        Compute the loss and weight gradients for a batch of examples.

        Parameters
        ----------
        input_vector : np.ndarray
            Input vector to be forwarded into the neural network.
        target_vector : np.ndarray
            Target vector for the training example.

        Returns
        -------
        loss : float
            Loss computed for the current batch.
        weights_grad : list of numpy.ndarray
            Weight gradient matrices.
        """
        self.training_state.previous_weights_deltas = update_weights(
            weights=self.weights,
            weights_gradients=self.training_state.weights_gradients,
            previous_weights_deltas=self.training_state.previous_weights_deltas,
            update_params=self.update_params
        )

    def _training_step(
            self,
            input_vector: np.ndarray,
            target_vector: np.ndarray,
            valid_input_vector: np.ndarray | None,
            valid_target_vector: np.ndarray | None
            ) -> None:

        # Compute training loss
        training_loss = self.loss_wrapper(
            input_vector=input_vector,
            target_vector=target_vector
        )
        self.training_log.loss.append(training_loss)

        # Compute gradients on training data
        if (
            self.training_params.batch_size is None
            or self.stopping_crit.min_gradient_norm is not None
        ):
            self._backprop_wrapper()
            self.training_state.gradient_norms.append(
                compute_gradient_norm(self.training_state.weights_gradients)
            )

        # Evaluate validation loss before updating weights
        if self.stopping_crit.min_delta_valid_loss is not None:
            validation_loss = self.loss_wrapper(
                input_vector=valid_input_vector,
                target_vector=valid_target_vector
            )
            self.training_log.validation_loss.append(validation_loss)
            self._update_early_stopping(validation_loss)

        # Update weights
        if self.training_params.batch_size is None:
            self._update_wrapper()
        else:
            self._training_step_online(
                input_vector=input_vector,
                target_vector=target_vector
            )

    def _training_step_online(
            self,
            input_vector: np.ndarray,
            target_vector: np.ndarray
            ) -> None:
        """
        Update the network weights using mini-batch gradient descent.

        The input and target datasets are divided into smaller batches.
        Each batch is processed independently, and the weights are
        updated after processing it.

        Parameters
        ----------
        input_vector : np.ndarray
            Input vector to be forwarded into the neural network.
        target_vector : np.ndarray
            Target vector for the training example.
        """
        batch_size = self.training_params.batch_size
        # Split training inputs/targets sets into sub-arrays based on batch_size
        mini_batch_input = np.array_split(
            input_vector,
            np.arange(batch_size, input_vector.shape[1], batch_size),
            axis=1
        )
        mini_batch_target = np.array_split(
            target_vector,
            np.arange(batch_size, target_vector.shape[1], batch_size),
            axis=1
        )
        # Perform forwarding, back-prop and weights update for each mini-batch
        for batch_input, batch_target in zip(mini_batch_input, mini_batch_target):
            output, net = forward_propagation(
                weights=self.weights,
                act_funcs=self.architecture.act_funcs,
                input_vector=batch_input
            )

            error = compute_error(
                output_vectors=output,
                target_vector=batch_target
            )

            weights_grad = backward_propagation(
                weights=self.weights,
                act_funcs=self.architecture.act_funcs,
                output_vectors=output,
                computed_nets=net,
                error_vector=error
            )

            self.training_state.previous_weights_deltas = update_weights(
                weights=self.weights,
                weights_gradients=weights_grad,
                previous_weights_deltas=self.training_state.previous_weights_deltas,
                update_params=self.update_params
            )

    def training_loop(
            self,
            input_vector: np.ndarray,
            target_vector: np.ndarray,
            valid_input_vector: np.ndarray | None = None,
            valid_target_vector: np.ndarray | None = None
            ) -> TrainingLog:
        """
        Train the neural network until a stopping criterion is satisfied.

        At each epoch, the method performs a training step, records the
        loss and gradient norm, and checks the configured stopping
        criteria.

        Parameters
        ----------
        input_vector : np.ndarray
            Input vector to be forwarded into the neural network.
        target_vector : np.ndarray
            Target vector for the training example.
        valid_input_vector : np.ndarray or None, default=None
            Validation set input vector used to provide early stopping criteria.
        valid_target_vector : np.ndarray or None, default=None
            Validation set target vector used to provide early stopping criteria.

        Returns
        -------
        TrainingLog
            Training history containing the random seed, loss values,
            gradient norms, elapsed epochs, stopping criteria and
            execution timestamps.
        """

        self.training_log.training_start_stop=[datetime.now()]

        # Start training loop until a stopping condition is met
        while not self.training_log.stopping_cause:
            # Execute a training step
            self._training_step(
                input_vector=input_vector,
                target_vector=target_vector,
                valid_input_vector=valid_input_vector,
                valid_target_vector=valid_target_vector
            )

            # Check stopping criteria
            self._check_stopping_criteria()

        # Restore the weights corresponding to the lowest validation loss
        if (
            self.stopping_crit.min_delta_valid_loss is not None
            and self.es_state.best_weights is not None
        ):
            for weights, best_weights in zip(
                self.weights,
                self.es_state.best_weights
            ):
                weights[...] = best_weights

            self.training_log.best_epoch = self.es_state.best_epoch

        # When training is completed dump the remaining
        # training data into a training log and return it
        self.training_log.training_start_stop.append(datetime.now())
        self.training_log.last_delta_loss = list(np.abs(np.diff(self.training_log.loss)))
        self.training_log.training_state = self.training_state

        return self.training_log
