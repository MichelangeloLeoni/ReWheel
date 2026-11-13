"""nn_simulator main module. Contains the NeuralNetwork class."""
import os
from datetime import datetime
import numpy as np
from .classes import (
    TrainingParameters,
    UpdateParameters,
    Architecture,
    TrainingLog,
    StoppingCriteria
)
from .propagation import forward_propagation
from .utils import check_input
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
        """
        Return a string representation of the neural network.

        The representation includes the network architecture, weight
        initialization status and any available training and update
        parameters or training log.

        Returns
        -------
        str
            Readable representation of the neural network instance.
        """
        weights_initialized = all(
            w is not None for w in self.weights
        )
        weights_status = (
            "initialized"
            if weights_initialized
            else "uninitialized"
        )
        lines = [
            "NeuralNetwork(",
            f"    architecture={self.architecture},",
            f"    weights={weights_status},",
        ]
        if self.training_params is not None:
            lines.append(
                f"    training_params={self.training_params},"
            )
        if self.update_params is not None:
            lines.append(
                f"    update_params={self.update_params},"
            )
        if self.training_log is not None:
            lines.append(
                f"    training_log={self.training_log},"
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

    def save_model(
            self,
            model_name: str = None,
            models_dir: str = "models"
            ) -> str:
        """
        Save the neural network model to a directory.

        The model is saved as a collection of NumPy files containing
        the network architecture, weight matrices and available
        training configuration.

        If no model name is provided, a name based on the current
        date and time is generated.

        Parameters
        ----------
        model_name : str or None, default=None
            Name of the directory where the model will be saved.
            If None, a timestamp-based name is generated.
        models_dir : str, default="models"
            Parent directory where the model directory will be created.
            It is created if it does not already exist.

        Returns
        -------
        str
            Path to the directory containing the saved model.

        Notes
        -----
        The following files are saved:

        - ``weights.npy``: network weight matrices.
        - ``architecture.npy``: network architecture.
        - ``training_parameters.npy``: training parameters, if available.
        - ``update_parameters.npy``: weight update parameters, if available.
        - ``stopping_criteria.npy``: stopping criteria, if available.
        - ``training_log.npy``: training history, if available.
        """
        # Generate model name if not provided
        if model_name is None:
            model_name = (
                f"model_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )

        # Create model directory
        model_path = os.path.join(models_dir, model_name)
        os.makedirs(model_path, exist_ok=True)

        # Save weights
        np.save(
            os.path.join(model_path, "weights.npy"),
            np.array(self.weights, dtype=object),
            allow_pickle=True
        )

        # Save architecture
        np.save(
            os.path.join(model_path, "architecture.npy"),
            self.architecture,
            allow_pickle=True
        )

        # Save training parameters
        if self.training_params is not None:
            np.save(
                os.path.join(model_path, "training_parameters.npy"),
                self.training_params,
                allow_pickle=True
            )

        # Save update parameters
        if self.update_params is not None:
            np.save(
                os.path.join(model_path, "update_parameters.npy"),
                self.update_params,
                allow_pickle=True
            )

        # Save stopping criteria
        if self.stopping_crit is not None:
            np.save(
                os.path.join(model_path, "stopping_criteria.npy"),
                self.stopping_crit,
                allow_pickle=True
            )

        # Save training log
        if self.training_log is not None:
            np.save(
                os.path.join(model_path, "training_log.npy"),
                self.training_log,
                allow_pickle=True
            )

        return model_path

    @classmethod
    def load_model(
            cls,
            model_path: str
            ) -> "NeuralNetwork":
        """
        Load a neural network model from a directory.

        The method restores the network architecture, weight matrices
        and available training configuration.

        Parameters
        ----------
        model_path : str
            Path to the directory containing the saved model files.

        Returns
        -------
        NeuralNetwork
            New neural network instance initialized with the loaded
            architecture, weights and available parameters.

        Notes
        -----
        The following files are required:

        - ``architecture.npy``: network architecture.
        - ``weights.npy``: network weight matrices.

        Training, update and stopping parameters are restored only
        if their corresponding files are present.
        """
        # Load architecture
        architecture = np.load(
            os.path.join(model_path, "architecture.npy"),
            allow_pickle=True
        ).item()

        # Initialize neural network
        nn = cls(architecture=architecture)

        # Load weights
        nn.weights = np.load(
            os.path.join(model_path, "weights.npy"),
            allow_pickle=True
        ).tolist()

        # Define optional parameter files
        parameter_files = {
            "training_params": "training_parameters.npy",
            "update_params": "update_parameters.npy",
            "stopping_crit": "stopping_criteria.npy"
        }

        # Load available training configuration
        for attribute, filename in parameter_files.items():
            parameter_path = os.path.join(model_path, filename)
            if os.path.isfile(parameter_path):
                setattr(
                    nn,
                    attribute,
                    np.load(
                        parameter_path,
                        allow_pickle=True
                    ).item()
                )

        return nn

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

        # Check if the input vector is suitable
        check_input(
            input_vector=input_vector,
            input_vector_size=self.architecture.input_vector_size
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
        attribute. The training log is also printed after training
        is completed.
        """
        # Check if the input vector is suitable
        check_input(
            input_vector=input_vector,
            input_vector_size=self.architecture.input_vector_size
        )

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
            architecture=self.architecture,
            update_params=self.update_params,
            training_params=self.training_params,
            stopping_crit=self.stopping_crit,
            weights=self.weights
        )

        # Start training loop
        self.training_log = tr.training_loop(
            input_vector=input_vector,
            target_vector=target_vector,
            valid_input_vector=valid_input_vector,
            valid_target_vector=valid_target_vector
        )

        # Print training log
        self.log()

    def log(self) -> None:
        """
        Print the training log.

        If a training log is available, its string representation
        is printed.
        """
        if self.training_log is not None:
            print(self.training_log)
