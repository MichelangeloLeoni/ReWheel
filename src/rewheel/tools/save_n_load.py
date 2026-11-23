"""
Module providing utilities to save and load neural network models.
"""
import os
from datetime import datetime
import numpy as np
from ..engine.nn_sim import NeuralNetwork

def save_model(
        nn: NeuralNetwork,
        model_name: str = None,
        models_dir: str = "models"
        ) -> str:
    """
    Save a neural network model to disk.

    The model is saved in a dedicated directory containing NumPy
    files with the network architecture, weight matrices and
    available training configuration.

    If no model name is provided, a name based on the current
    date and time is generated.

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
        np.array(nn.weights, dtype=object),
        allow_pickle=True
    )

    # Save architecture
    np.save(
        os.path.join(model_path, "architecture.npy"),
        nn.architecture,
        allow_pickle=True
    )

    # Save training parameters
    if nn.training_params is not None:
        np.save(
            os.path.join(model_path, "training_parameters.npy"),
            nn.training_params,
            allow_pickle=True
        )

    # Save update parameters
    if nn.update_params is not None:
        np.save(
            os.path.join(model_path, "update_parameters.npy"),
            nn.update_params,
            allow_pickle=True
        )

    # Save stopping criteria
    if nn.stopping_crit is not None:
        np.save(
            os.path.join(model_path, "stopping_criteria.npy"),
            nn.stopping_crit,
            allow_pickle=True
        )

    # Save training log
    if nn.training_log is not None:
        np.save(
            os.path.join(model_path, "training_log.npy"),
            nn.training_log,
            allow_pickle=True
        )

    return model_path

def load_model(
        model_path: str
        ) -> NeuralNetwork:
    """
    Load a neural network model from disk.

    The function restores the network architecture, weight matrices
    and available training configuration, returning a new
    NeuralNetwork instance.

    Parameters
    ----------
    model_path : str
        Path to the directory containing the saved model files.

    Returns
    -------
    NeuralNetwork
        Restored neural network instance initialized with the
        loaded architecture, weights and available parameters.

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
    nn = NeuralNetwork(architecture=architecture)

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
