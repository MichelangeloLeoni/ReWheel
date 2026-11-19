"""Module containing functions to manage neural network propagation and updates during training."""
from typing import TYPE_CHECKING
import numpy as np
from ..propagation import forward_propagation, backward_propagation, update_weights
from ..loss import compute_error, compute_loss

# NeuralNetwork is only imported during type checking and
# ignored during runtime, preventing circular imports while
# allowing type hinting
if TYPE_CHECKING:
    from ..nn_sim import NeuralNetwork

def propagate(
        nn: "NeuralNetwork",
        input_vector: np.ndarray,
        target_vector: np.ndarray
        ) -> tuple[list[np.ndarray], float]:
    """
    Performs forward and backward propagation on a set of training samples.

    Parameters
    ----------
    nn : NeuralNetwork
        Neural network to propagate through.
    input_vector : np.ndarray
        Input samples, arranged in columns.
    target_vector : np.ndarray
        Target values corresponding to the input samples.

    Returns
    -------
    tuple[list[np.ndarray], float]
        A tuple containing the weight gradients and the loss computed
        over the input samples.
    """
    outputs, nets = forward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        input_vector=input_vector
    )
    error = compute_error(
        output_vectors=outputs,
        target_vector=target_vector
    )
    weights_gradients = backward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        output_vectors=outputs,
        computed_nets=nets,
        error_vector=error
    )
    loss = compute_loss(
        output_vectors=outputs,
        target_vector=target_vector
    )

    return weights_gradients, loss

def propagate_valid(
        nn: "NeuralNetwork",
        input_vector: np.ndarray,
        target_vector: np.ndarray
        ) -> float:
    """
    Computes the loss of a set of validation samples.

    Performs forward propagation without computing gradients or updating
    the network weights.

    Parameters
    ----------
    nn : NeuralNetwork
        Neural network to evaluate.
    input_vector : np.ndarray
        Validation samples, arranged in columns.
    target_vector : np.ndarray
        Target values corresponding to the validation samples.

    Returns
    -------
    float
        Loss computed over the validation samples.
    """
    outputs, _ = forward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        input_vector=input_vector
    )
    loss = compute_loss(
        output_vectors=outputs,
        target_vector=target_vector
    )

    return loss

def update(
        nn: "NeuralNetwork",
        weights_gradients: list[np.ndarray],
        previous_weights_deltas: list[np.ndarray],
        input_vector: np.ndarray,
        target_vector: np.ndarray,
        seed: int
        ) -> list[np.ndarray]:
    """
    Updates the neural network weights using batch or mini-batch training.

    When batch training is used, the supplied gradients are directly
    applied to the network weights. When mini-batch training is used,
    the training samples are shuffled, divided into mini-batches and
    processed sequentially.

    Parameters
    ----------
    nn : NeuralNetwork
        Neural network whose weights are updated.
    weights_gradients : list[np.ndarray]
        Gradients of the network weights computed over the training samples.
        Used only when batch training is enabled.
    previous_weights_deltas : list[np.ndarray]
        Weight updates from the previous training step, used to compute
        momentum-based updates.
    input_vector : np.ndarray
        Training samples, arranged in columns.
    target_vector : np.ndarray
        Target values corresponding to the training samples.
    seed : int
        Seed used to initialize the random number generator for shuffling
        the training samples.

    Returns
    -------
    list[np.ndarray]
        Weight updates from the last update step, to be used in the
        subsequent training step.
    """
    batch_size = nn.training_params.batch_size

    # If we are batch-training, apply the supplied gradients directly
    if batch_size is None:
        previous_weights_deltas = update_weights(
            weights=nn.weights,
            weights_gradients=weights_gradients,
            previous_weights_deltas=previous_weights_deltas,
            update_params=nn.update_params
        )

    # If we are mini-batch-training, split the samples into mini-batches
    # and perform a separate training step for each mini-batch
    else:
        # Shuffle training samples
        rng = np.random.default_rng(seed)
        indices = rng.permutation(input_vector.shape[1])
        input_vector = input_vector[:, indices]
        target_vector = target_vector[:, indices]

        # Split input and target samples into mini-batches
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

        # Perform forward propagation, backpropagation and weight updates
        # for each mini-batch
        for batch_input, batch_target in zip(mini_batch_input, mini_batch_target):
            output, net = forward_propagation(
                weights=nn.weights,
                act_funcs=nn.architecture.act_funcs,
                input_vector=batch_input
            )
            error = compute_error(
                output_vectors=output,
                target_vector=batch_target
            )
            weights_grad = backward_propagation(
                weights=nn.weights,
                act_funcs=nn.architecture.act_funcs,
                output_vectors=output,
                computed_nets=net,
                error_vector=error
            )
            previous_weights_deltas = update_weights(
                weights=nn.weights,
                weights_gradients=weights_grad,
                previous_weights_deltas=previous_weights_deltas,
                update_params=nn.update_params
            )

    return previous_weights_deltas
