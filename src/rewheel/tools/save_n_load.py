"""
Module providing utilities to save and load neural network models.
"""
import os
from contextlib import redirect_stdout
from datetime import datetime
import numpy as np
from ..engine.nn_sim import NeuralNetwork

def _generate_metadata(
        data: object,
        model_path: str,
        ) -> None:
    """
    Helper function that writes the string representation
    of input data to a file.

    Needed to generate a human-readable file that contains
    neural network metadata.

    Parameters
    ----------
    data : object
        Any data to be string represented inside the file.
    model_path : str
        Name of the directory where the file will be saved.
    """
    file_path = os.path.join(model_path, "metadata.txt")
    with open(file_path, "a", encoding="utf-8") as file:
        with redirect_stdout(file):
            print(data)
            print()

def save_model(
        nn: NeuralNetwork,
        model_name: str = None,
        models_dir: str = "models"
        ) -> str:
    """
    Save a neural network model to disk.

    The model is saved in a dedicated directory containing NumPy
    files with the network architecture, weight matrices, training
    configuration and training history, when available.

    If no model name is provided, a name based on the current date
    and time is generated.

    Parameters
    ----------
    nn : NeuralNetwork
        Neural network instance to be saved.
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

    - ``weights.npy`` : Network weight matrices.
    - ``architecture.npy`` : Network architecture.
    - ``loss.npy`` : Loss function associated with the network.
    - ``training_parameters.npy`` : Training parameters, if available.
    - ``update_parameters.npy`` : Weight update parameters, if available.
    - ``stopping_criteria.npy`` : Training stopping criteria, if available.
    - ``training_log.npy`` : Training history, if available.
    - ``metadata.txt`` : Human-readable representations of the
      saved network architecture, loss function and available
      training configuration and history.

    The training log contains information collected during training,
    such as loss history, validation loss, stopping cause, best epoch,
    random seed and training state.

    Files are saved using NumPy's binary format, while the metadata
    file provides a human-readable summary of the saved objects.
    """
    # Generate model name if not provided
    if model_name is None:
        model_name = (
            f"model_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

    # Create model directory
    model_path = os.path.join(models_dir, model_name)
    os.makedirs(model_path, exist_ok=True)

    # Clear metadata file
    metadata_path = os.path.join(model_path, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8"):
        pass

    # Save weights
    np.save(
        os.path.join(model_path, "weights.npy"),
        np.array(nn.weights, dtype=object),
        allow_pickle=True
    )

    # Save architecture
    np.save(
        os.path.join(model_path, "architecture.npy"),
        nn.architecture,
        allow_pickle=True
    )
    _generate_metadata(data=nn.architecture, model_path=model_path)

    # Save training loss
    np.save(
        os.path.join(model_path, "loss.npy"),
        nn.loss,
        allow_pickle=True
    )
    _generate_metadata(data=nn.loss, model_path=model_path)

    # Save training parameters
    if nn.training_params is not None:
        np.save(
            os.path.join(model_path, "training_parameters.npy"),
            nn.training_params,
            allow_pickle=True
        )
        _generate_metadata(data=nn.training_params, model_path=model_path)

    # Save update parameters
    if nn.update_params is not None:
        np.save(
            os.path.join(model_path, "update_parameters.npy"),
            nn.update_params,
            allow_pickle=True
        )
        _generate_metadata(data=nn.update_params, model_path=model_path)

    # Save stopping criteria
    if nn.stopping_crit is not None:
        np.save(
            os.path.join(model_path, "stopping_criteria.npy"),
            nn.stopping_crit,
            allow_pickle=True
        )
        _generate_metadata(data=nn.stopping_crit, model_path=model_path)

    # Save training log
    if nn.training_log is not None:
        np.save(
            os.path.join(model_path, "training_log.npy"),
            nn.training_log,
            allow_pickle=True
        )
        _generate_metadata(data=nn.training_log, model_path=model_path)

    return model_path

def load_model(
        model_path: str
        ) -> NeuralNetwork:
    """
    Load a neural network model from disk.

    The function restores a neural network from a previously saved
    model directory, including its architecture, weight matrices,
    loss function and available training configuration and history.

    Parameters
    ----------
    model_path : str
        Path to the directory containing the saved model files.

    Returns
    -------
    NeuralNetwork
        Restored neural network instance initialized with the
        loaded architecture, weights, loss function and available
        training information.

    Notes
    -----
    The following files are required:

    - ``architecture.npy`` : Network architecture.
    - ``weights.npy`` : Network weight matrices.
    - ``loss.npy`` : Loss function associated with the network.

    The following files are optional and are loaded only if present:

    - ``training_parameters.npy`` : Training parameters.
    - ``update_parameters.npy`` : Weight update parameters.
    - ``stopping_criteria.npy`` : Training stopping criteria.
    - ``training_log.npy`` : Training history.

    The restored training log provides access to the information
    collected during the original training process, including loss
    history, validation loss, stopping cause, best epoch, random
    seed and training state.

    Notes
    -----
    Files are loaded using ``allow_pickle=True`` because the saved
    objects include custom Python classes and object arrays.
    """
    # Load architecture
    architecture = np.load(
        os.path.join(model_path, "architecture.npy"),
        allow_pickle=True
    ).item()

    # Load training loss
    loss = np.load(
        os.path.join(model_path, "loss.npy"),
        allow_pickle=True
    ).item()

    # Initialize neural network
    nn = NeuralNetwork(
        architecture=architecture,
        loss=loss
    )

    # Load weights
    nn.weights = np.load(
        os.path.join(model_path, "weights.npy"),
        allow_pickle=True
    ).tolist()

    # Define optional parameter files
    parameter_files = {
        "training_params": "training_parameters.npy",
        "update_params": "update_parameters.npy",
        "stopping_crit": "stopping_criteria.npy",
        "training_log": "training_log.npy"
    }

    # Restore available training configuration
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
