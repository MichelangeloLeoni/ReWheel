"""
Module containing functions to manage neural network propagation
and weight updates during training.
"""
from typing import TYPE_CHECKING
import numpy as np
from ..propagation import (
    forward_propagation,
    backward_propagation,
    update_weights
)
from ..loss import compute_loss
from ..delta import compute_output_delta

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

    Computes the network outputs through forward propagation, evaluates
    the prediction errors, and calculates the weight gradients through
    backward propagation. It also computes the loss over the training
    samples.

    No weight updates are performed.

    Parameters
    ----------
    nn : NeuralNetwork
        Neural network used for propagation.
    input_vector : np.ndarray
        Training input samples, arranged in columns.
    target_vector : np.ndarray
        Target values corresponding to the training samples,
        arranged in columns.

    Returns
    -------
    tuple[list[np.ndarray], float]
        A tuple containing:
        - weights_gradients : list of np.ndarray
            Gradients of the loss with respect to the network weights.
        - loss : float
            Loss computed over the training samples.
    """
    results = forward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        input_vector=input_vector
    )
    output_delta = compute_output_delta(
        forward_results=results,
        target_vector=target_vector,
        act_func=nn.architecture.act_funcs,
        loss=nn.loss
    )
    weights_gradients = backward_propagation(
        forward_results=results,
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        output_delta=output_delta
    )
    loss = compute_loss(
        output_vectors=results.output_vectors,
        target_vector=target_vector,
        loss_function=nn.loss
    )

    return weights_gradients, loss

def propagate_valid(
        nn: "NeuralNetwork",
        input_vector: np.ndarray,
        target_vector: np.ndarray
        ) -> float:
    """
    Computes the loss over a set of validation samples.

    Performs forward propagation to obtain the network predictions
    and evaluates the corresponding loss. No gradients are computed
    and the network weights remain unchanged.

    Parameters
    ----------
    nn : NeuralNetwork
        Neural network used for validation.
    input_vector : np.ndarray
        Validation input samples, arranged in columns.
    target_vector : np.ndarray
        Target values corresponding to the validation samples,
        arranged in columns.

    Returns
    -------
    float
        Loss computed over the validation samples.
    """
    results = forward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        input_vector=input_vector
    )
    loss = compute_loss(
        output_vectors=results.output_vectors,
        target_vector=target_vector,
        loss_function=nn.loss
    )

    return loss

def update_mini_batches(
        nn: "NeuralNetwork",
        previous_weights_deltas: list[np.ndarray],
        input_vector: np.ndarray,
        target_vector: np.ndarray,
        seed: int | None = None
        ) -> list[np.ndarray]:
    """
    Updates the network weights using shuffled mini-batches.

    Shuffles the training samples, divides them into mini-batches
    according to the configured batch size, and performs forward
    propagation, backward propagation, and weight updates for each
    mini-batch.

    Weight deltas are retained between consecutive mini-batches
    to support momentum-based updates.

    Parameters
    ----------
    nn : NeuralNetwork
        Neural network whose weights are updated.
    previous_weights_deltas : list of np.ndarray
        Weight deltas from the previous update step, used to compute
        momentum-based updates.
    input_vector : np.ndarray
        Training input samples, arranged in columns.
    target_vector : np.ndarray
        Target values corresponding to the training samples,
        arranged in columns.
    seed : int or None, default=None
        Random seed used to shuffle the training samples.

        If an integer is provided, the same permutation is generated
        every time the function is called with that seed.

        If None, a new random seed is generated at each call,
        resulting in a different permutation.

    Returns
    -------
    list of np.ndarray
        Weight deltas resulting from the final mini-batch update.
        These are used as the previous weight deltas in the next
        training iteration.

    Notes
    -----
    The batch size is retrieved from the neural network training
    parameters.

    The final mini-batch may contain fewer samples than the configured
    batch size if the number of training samples is not divisible
    by the batch size.

    Providing a fixed seed ensures reproducible shuffling across
    function calls, while using None generates independent shuffling
    sequences across calls.
    """
    batch_size = nn.training_params.batch_size

    # If no seed is provided generate a different seed for each
    # mini-batch training cycle
    if seed is None:
        seed = np.random.SeedSequence().generate_state(
            1, dtype=np.uint32
        )[0].item()

    # Shuffle training samples
    indices = np.random.default_rng(seed).permutation(input_vector.shape[1])
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

    # Perform updates for each mini-batch
    for batch_input, batch_target in zip(
            mini_batch_input,
            mini_batch_target
            ):
        results = forward_propagation(
            weights=nn.weights,
            act_funcs=nn.architecture.act_funcs,
            input_vector=batch_input
        )
        output_delta = compute_output_delta(
            forward_results=results,
            target_vector=batch_target,
            act_func=nn.architecture.act_funcs,
            loss=nn.loss
        )
        weights_grad = backward_propagation(
            forward_results=results,
            weights=nn.weights,
            act_funcs=nn.architecture.act_funcs,
            output_delta=output_delta
        )
        previous_weights_deltas = update_weights(
            weights=nn.weights,
            weights_gradients=weights_grad,
            previous_weights_deltas=previous_weights_deltas,
            update_params=nn.update_params
        )

    return previous_weights_deltas
