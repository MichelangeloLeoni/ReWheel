"""Module containing functions used to evaluate neural network predictions."""
import numpy as np
from ..engine.nn_sim import NeuralNetwork
from ..engine.utils import check_matching_samples, check_matching_features

def _get_classification_type(
        network: NeuralNetwork
        ) -> str:
    """
    Determine the classification type from a neural network architecture.

    The classification type is inferred from the number of output units
    and the activation function of the output layer.

    Parameters
    ----------
    network : NeuralNetwork
        Neural network whose output configuration is inspected.

    Returns
    -------
    str
        Classification type:
        - 'B' for binary classification.
        - 'ML' for multilabel classification.
        - 'MC' for multiclass classification.

    Raises
    ------
    ValueError
        If the output configuration is not supported for classification.
    """
    output_units = network.architecture.layers[-1]
    out_act = network.architecture.act_funcs[-1]

    if output_units == 1 and out_act == 'SIGMOID':
        return 'B'
    if out_act == 'SIGMOID':
        return 'ML'
    if out_act == 'SOFTMAX':
        return 'MC'

    raise ValueError(
        "Unsupported output configuration for classification."
    )

def _infer_classification_type(
        target_vector: np.ndarray
        ) -> str:
    """
    Infer the classification type from the target data.

    The classification type is determined from the target values and
    their arrangement. Targets must use binary or one-hot encoding.

    Parameters
    ----------
    target_vector : np.ndarray
        Target data. Expected to have shape (n_features, n_samples)
        or be a one-dimensional array.

    Returns
    -------
    str
        Inferred classification type:
        - 'B' for binary classification.
        - 'ML' for multilabel classification.
        - 'MC' for multiclass classification.

    Raises
    ------
    ValueError
        If the target data contains values other than 0 and 1.
    """
    if not np.all(np.isin(target_vector, [0, 1])):
        raise ValueError(
            "Cannot infer classification type from target data."
        )

    if target_vector.ndim == 1:
        if target_vector.size == 1:
            return 'B'
        if np.sum(target_vector) == 1:
            return 'MC'
        return 'ML'

    if target_vector.shape[0] == 1:
        return 'B'

    active_labels = np.sum(target_vector, axis=0)
    if np.all(active_labels == 1):
        return 'MC'

    return 'ML'

def _compute_confusion_matrix(
        true_classes: np.ndarray,
        predicted_classes: np.ndarray,
        n_classes: int
        ) -> np.ndarray:
    """
    Compute the confusion matrix from true and predicted class labels.

    Rows represent true classes, while columns represent predicted
    classes. Each element indicates the number of samples assigned
    to the corresponding true-predicted class pair.

    Parameters
    ----------
    true_classes : np.ndarray
        Array containing the true class indices.
    predicted_classes : np.ndarray
        Array containing the predicted class indices.
    n_classes : int
        Number of classes. Determines the dimensions of the
        resulting confusion matrix.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes), where
        element [i, j] represents the number of samples belonging
        to class i and predicted as class j.
    """
    true_classes = np.asarray(true_classes).ravel()
    predicted_classes = np.asarray(predicted_classes).ravel()

    confusion_matrix = np.zeros(
        (n_classes, n_classes),
        dtype=int
    )

    for true_class, predicted_class in zip(
            true_classes, predicted_classes
            ):
        confusion_matrix[true_class, predicted_class] += 1

    return confusion_matrix

def evaluate(
        output_vector: np.ndarray,
        target_vector: np.ndarray,
        threshold: float = 0.5,
        network: NeuralNetwork | None = None
        ) -> tuple[float, np.ndarray]:
    """
    Evaluate neural network predictions using classification accuracy.

    Supports binary, multiclass and multilabel classification.
    The classification type is inferred from the target data when
    no neural network is provided, otherwise it is determined from
    the network output architecture.

    Binary and multilabel predictions are converted into class labels
    using the specified threshold. Multiclass predictions are assigned
    to the class with the highest output value.

    Parameters
    ----------
    output_vector : np.ndarray
        Neural network predictions, with shape (n_features, n_samples).
    target_vector : np.ndarray
        Target data, with shape (n_features, n_samples).
        Binary and multilabel targets must contain only 0 and 1.
        Multiclass targets must use one-hot encoding.
    threshold : float, default=0.5
        Threshold used to convert binary and multilabel predictions
        into class labels. Must be between 0 and 1.
    network : NeuralNetwork or None, default=None
        Neural network used to determine the classification type.
        If None, the classification type is inferred from the
        target data.

    Returns
    -------
    tuple[float, np.ndarray]
        A tuple containing:
        - accuracy : float
            Classification accuracy. For binary and multiclass
            classification, this is the fraction of correctly
            classified samples. For multilabel classification,
            a sample is considered correct only if all its labels
            are predicted correctly.
        - confusion_matrix : np.ndarray
            Confusion matrix. For binary and multiclass classification,
            has shape (n_classes, n_classes). For multilabel
            classification, has shape (n_features, 2, 2), containing
            one confusion matrix for each label.

    Raises
    ------
    ValueError
        If the number of features or samples differs between
        predictions and targets.
    ValueError
        If the threshold is outside the interval [0, 1].
    ValueError
        If the classification type cannot be determined.
    ValueError
        If target data does not satisfy the encoding requirements
        of the selected classification type.
    
    Notes
    -----
    Once the accuracy is computed it is printed in the console.
    """
    # Check if the input vectors type is np.ndarray
    if not isinstance(output_vector, np.ndarray):
        output_vector = np.asarray(output_vector)
    if not isinstance(target_vector, np.ndarray):
        target_vector = np.asarray(target_vector)

    # Check if input vectors number of features and examples match
    check_matching_features(output_vector, target_vector)
    check_matching_samples(output_vector, target_vector)

    # Check if the threshold value is between 1 and 0
    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    # If the neural network is provided get the classification type
    # from the architecture, otherwise try to infer it from the inputs
    if network is None:
        class_type = _infer_classification_type(target_vector)
    else:
        class_type = _get_classification_type(network)

    # Multiclass
    if class_type == 'MC':
        if not np.all(np.isin(target_vector, [0, 1])):
            raise ValueError(
                "Multiclass targets must use one-hot encoding."
            )
        if not np.all(np.sum(target_vector, axis=0) == 1):
            raise ValueError(
                "Each multiclass target must contain exactly "
                "one active class."
            )

        # The predicted class corresponds to the highest output value
        predicted_classes = np.argmax(output_vector, axis=0)
        true_classes = np.argmax(target_vector, axis=0)

        # Number of class = number of output features
        n_classes = output_vector.shape[0]

        # Compute confusin matrix and accuracy
        confusion_matrix = _compute_confusion_matrix(
            true_classes,
            predicted_classes,
            n_classes
        )
        accuracy = np.mean(
            predicted_classes == true_classes
        )

    # Binary
    elif class_type == 'B':
        if not np.all(np.isin(target_vector, [0, 1])):
            raise ValueError(
                "Binary targets must contain only 0 and 1."
            )

        # The predicted class corresponds to the value above threshold
        predicted_classes = (
            output_vector >= threshold
        ).astype(int).ravel()
        true_classes = target_vector.ravel().astype(int)

        # Compute confusin matrix and accuracy
        confusion_matrix = _compute_confusion_matrix(
            true_classes,
            predicted_classes,
            2
        )
        accuracy = np.mean(
            predicted_classes == true_classes
        )

    # Multilabel
    else:
        if not np.all(np.isin(target_vector, [0, 1])):
            raise ValueError(
                "Multilabel targets must contain only 0 and 1."
            )

        # The predicted class corresponds to the value above threshold
        predicted_classes = (
            output_vector >= threshold
        ).astype(int)
        true_classes = target_vector.astype(int)

        # Number of class = number of output features
        n_classes = output_vector.shape[0]

        # Compute confusin matrix and accuracy
        confusion_matrix = np.stack([
            _compute_confusion_matrix(
                true_classes[i],
                predicted_classes[i],
                2
            )
            for i in range(n_classes)
        ])
        accuracy = np.mean(
            np.all(
                predicted_classes == true_classes,
                axis=0
            )
        )

    print(f"{accuracy:.3f}%")

    return float(accuracy), confusion_matrix
