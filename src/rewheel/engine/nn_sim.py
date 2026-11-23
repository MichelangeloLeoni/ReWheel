"""nn_simulator main module. Contains the NeuralNetwork class."""
import numpy as np
from .classes import (
    TrainingParameters,
    UpdateParameters,
    Architecture,
    TrainingLog,
    StoppingCriteria
)
from .propagation import forward_propagation
from .utils import check_input, initialize_weights
from .training.trainer import Trainer

class NeuralNetwork:
    """
    Implement a feedforward neural network.

    This class provides the main interface for creating, training and
    using a neural network. It manages the network architecture, weight
    matrices, training and update parameters, and training history.

    The network supports saving and loading models, making predictions
    and training using the configured optimization parameters.

    Parameters
    ----------
    architecture : Architecture
        Architecture defining the number of neurons and activation
        functions in each layer.
    training_parameters : TrainingParameters, default=None
        Parameters controlling the training process and its stopping
        criteria. If None, training parameters must be provided when
        calling the train method.
    update_parameters : UpdateParameters, default=None
        Parameters controlling weight updates, including learning rate,
        momentum and regularization. If None, update parameters must be
        provided when calling the train method.
    stopping_criteria : StoppingCriteria or None, default=None
        Criteria determining when the training process terminates.
        If None, they must be provided before calling the train method.

    Attributes
    ----------
    architecture : Architecture
        Architecture of the neural network.
    weights : list of numpy.ndarray or None
        List of weight matrices connecting consecutive layers.
        Initialized during the first training process or when loading
        a model.
    update_params : UpdateParameters or None
        Parameters controlling weight updates.
    training_params : TrainingParameters or None
        Parameters controlling the training process.
    stopping_crit : StoppingCriteria or None
        Criteria determining when the training process terminates.
    training_log : TrainingLog or None
        Log containing the results and history of the latest training
        process.

    Notes
    -----
    The input and target datasets are expected to have examples
    arranged in columns.
    """

    def __init__(
            self,
            architecture: Architecture,
            training_parameters: TrainingParameters | None = None,
            update_parameters: UpdateParameters | None = None,
            stopping_criteria: StoppingCriteria | None = None
            ):
        # Initialize nn architecture attribute
        self.architecture = architecture
        # Initialize layer weights matrices list
        self.weights: list[np.ndarray | None] = [
            None
        ] * len(self.architecture.layers)
        # Initialize nn training/update parameters attributes
        self.update_params = update_parameters
        self.training_params = training_parameters
        self.stopping_crit = stopping_criteria
        # Initialize training log attribute to None
        self.training_log: TrainingLog = None

    def __repr__(self) -> str:
        weights_initialized = all(
            w is not None for w in self.weights
        )
        weights_status = (
            "initialized"
            if weights_initialized
            else "uninitialized"
        )

        nn_weights = [None] * len(self.architecture.layers)
        initialize_weights(self.architecture, nn_weights)
        total_params = sum(w.size for w in nn_weights)
        lines = [
            "NeuralNetwork(",
            f"    Architecture = {self.architecture},",
            f"    Weights status = {weights_status},",
            f"    Total trainable parameters = {total_params},"
        ]
        if self.training_params is not None:
            lines.append(
                f"    training_params = {self.training_params},"
            )
        if self.update_params is not None:
            lines.append(
                f"    update_params = {self.update_params},"
            )
        if self.stopping_crit is not None:
            lines.append(
                f"    stopping_crit = {self.stopping_crit},"
            )
        if self.training_log is not None:
            lines.append(
                f"    training_log = {self.training_log},"
            )
        lines.append(")")

        return "\n".join(lines)

    def configure(
            self,
            update_params: UpdateParameters | None = None,
            training_params: TrainingParameters | None = None,
            stopping_crit: StoppingCriteria | None = None
            ) -> None:
        """
        Configure the training and update parameters of the network.

        Only the parameters explicitly provided are updated. Parameters
        passed as None leave the corresponding attributes unchanged.

        Parameters
        ----------
        update_params : UpdateParameters or None, default=None
            Parameters controlling weight updates, including learning
            rate, momentum and regularization.
        training_params : TrainingParameters or None, default=None
            Parameters controlling the training process, including
            batch size and weight initialization settings.
        stopping_crit : StoppingCriteria or None, default=None
            Criteria determining when the training process terminates.
        """
        if update_params is not None:
            self.update_params = update_params
        if training_params is not None:
            self.training_params = training_params
        if stopping_crit is not None:
            self.stopping_crit = stopping_crit

    @check_input
    def predict(
            self,
            input_vector: np.ndarray
            ) -> np.ndarray:
        """
        Compute the neural network predictions.

        Performs forward propagation on the provided input examples
        and returns the output of the final network layer.

        Parameters
        ----------
        input_vector : np.ndarray
            Input vector to be forwarded into the neural network.

        Returns
        -------
        numpy.ndarray
            Predictions produced by the output layer. Each column
            corresponds to an input example.
        """
        # Checks if the model has been trained or loaded from
        # memory before attempting a prediction
        if all(w is None for w in self.weights):
            raise RuntimeError(
                "Cannot perform prediction: the model has not been trained or " \
                "loaded from memory."
            )

        # Performs forward propagation in the neural network
        output_vector = forward_propagation(
            weights=self.weights,
            act_funcs=self.architecture.act_funcs,
            input_vector=input_vector
        )[0][-1]

        # Returns only the output layer output vector
        # a.k.a. final prediction
        return output_vector

    @check_input
    def train(
            self,
            input_vector: np.ndarray,
            target_vector: np.ndarray,
            valid_input_vector: np.ndarray | None = None,
            valid_target_vector: np.ndarray | None = None
            ) -> None:
        """
        Train the neural network using the provided dataset.

        Initializes the network weights if necessary and performs
        training according to the configured training, update and
        stopping parameters.

        The training history is stored in the ``training_log``
        attribute after the training process is completed.

        Parameters
        ----------
        input_vector : np.ndarray
            Training input data, with features arranged in rows and
            examples arranged in columns.
        target_vector : np.ndarray
            Training target values, with output features arranged
            in rows and examples arranged in columns.
        valid_input_vector : np.ndarray or None, default=None
            Validation input data, with features arranged in rows
            and examples arranged in columns. Required when
            validation-loss-based early stopping is enabled.
        valid_target_vector : np.ndarray or None, default=None
            Validation target values, with output features arranged
            in rows and examples arranged in columns. Required when
            validation-loss-based early stopping is enabled.

        Raises
        ------
        ValueError
            If validation input data is provided without validation
            target data, or vice versa.
            If validation-loss-based early stopping is enabled but
            either validation dataset is missing.

        Notes
        -----
        The network weights are initialized by the Trainer if they
        have not already been initialized.

        The training results are stored in the ``training_log``
        attribute.
        """
        # Check if both validation vectors are passed to the function
        if self.stopping_crit.min_delta_valid_loss is not None:
            if valid_input_vector is None or valid_target_vector is None:
                raise ValueError(
                    "Both validation input vector and target vector must be passed during training."
                )
        # Check validation vectors type
        if valid_input_vector is not None and valid_target_vector is not None:
            if not isinstance(valid_input_vector, np.ndarray):
                valid_input_vector = np.array(valid_input_vector)
            if not isinstance(valid_target_vector, np.ndarray):
                valid_target_vector = np.array(valid_target_vector)

        # Initialize Trainer instance
        tr = Trainer(
            neural_network=self
        )

        # Start training loop
        self.training_log = tr.training_loop(
            input_vector=input_vector,
            target_vector=target_vector,
            valid_input_vector=valid_input_vector,
            valid_target_vector=valid_target_vector
        )
