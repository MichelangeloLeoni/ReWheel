"""
Module containing the functions to perform forward/backward propagation
and weights update in a neural network.
"""
import numpy as np
from . import activation_functions as af
from .delta import compute_delta
from .utils import add_bias, compute_gradient
from .classes import UpdateParameters, ForwardResults

def forward_propagation(
        weights: list[np.ndarray],
        act_funcs: list[str],
        input_vector: np.ndarray
        ) -> ForwardResults:
    """
    Performs the forwarding in a neural network.

    Parameters
    ----------
    weights : list of np.ndarray
        List of weights matrices of the neural network.
    act_funcs : list of str
        List of activation functions of each layer
        represented by a string.
    input_vector : np.ndarray
        Input vector to be forwarded into the neural network.

    Returns
    -------
    results : ForwardResults
        Object containing the intermediate results of the
        forward propagation:

            - output_vectors : list of np.ndarray
                Outputs of each layer, including the input layer.
                Bias terms are appended to the input and hidden layer
                outputs, but not to the output layer. The final element
                represents the network predictions.

            - computed_nets : list of np.ndarray
                Net inputs of each layer, computed before applying
                the corresponding activation function.
    """
    # Initialize ForwardResults instance
    num_layers = len(weights)
    results = ForwardResults(
        output_vectors = [None] * (num_layers + 1),
        computed_nets = [None] * num_layers
    )

    # Apply bias to the input vector
    # output_vector[0] represents the output of the input layer
    # that will be fed into the first hidden layer
    is_single_example = input_vector.ndim == 1
    results.output_vectors[0] = add_bias(input_vector, is_single_example)

    # Start propagating through layers
    for i, layer_weights in enumerate(weights):
        # Compute net of the whole layer
        net = layer_weights @ results.output_vectors[i]
        # Store the net of the layer
        results.computed_nets[i] = net

        # Apply activation function
        layer_output = af.af_list[act_funcs[i]](net)

        # Compute the output to be passed as input onto the next layer
        if i < num_layers-1:
            # The next layer input will be the biased version of the current layer output
            results.output_vectors[i+1] = add_bias(layer_output, is_single_example)
        else:
            # When we arrive at the output layer we save the output without adding a bias
            # This represents the neural network final prediction
            results.output_vectors[i+1] = layer_output

    return results

def backward_propagation(
        forward_results: ForwardResults,
        weights: list[np.ndarray],
        act_funcs: list[str],
        output_delta: np.ndarray
        ) -> list[np.ndarray]:
    """
    Performs the back-propagation in a neural network.
    Goes backward trough the layers comptuing their delta
    vectors and weights gradient matrices.

    Parameters
    ----------
    forward_results : ForwardResults
        Results of the forward propagation, containing the output
        vectors and net inputs computed for each layer.
    weights : list of np.ndarray
        List of weight matrices of the neural network.
    act_funcs : list of str
        List of activation functions used by each layer,
        represented by their corresponding names.
    output_delta : np.ndarray
        Delta vector of the output layer, computed from the network
        prediction, target values, loss function and output activation.

    Returns
    -------
    weights_gradients : list of np.ndarray
        List containing the negative weights gradient matrix of each layer.
    """
    num_layers = len(weights)
    is_single_example = forward_results.output_vectors[-1].ndim == 1

    # Initialize the weight gradients list
    weights_gradients = [None] * num_layers

    # Compute the output layer delta
    prev_delta = output_delta
    # Compute the output layer gradient
    weights_gradients[-1] = compute_gradient(
        delta=prev_delta,
        prev_layer_output=forward_results.output_vectors[-2],
        is_single_example=is_single_example
    )

    # Backpropagate through the hidden layers
    # We start looping from -2 since we already computed delta
    # and gradient for the output layer outside the loop
    for i in range(num_layers - 2, -1, -1):
        # Propagate the delta vector from the previous layer to the current
        propagated_error = (
            weights[i + 1][:, :-1].T @ prev_delta
        )

        # Compute delta vector for the current layer
        delta = compute_delta(
            propagated_error=propagated_error,
            act_func=act_funcs[i],
            computed_net=forward_results.computed_nets[i]
        )

        # Compute weights gradient for the current layer
        weights_gradients[i] = compute_gradient(
            delta=delta,
            prev_layer_output=forward_results.output_vectors[i],
            is_single_example=is_single_example
        )

        # Go to the previous layer
        prev_delta = delta

    return weights_gradients

def update_weights(
        weights: list[np.ndarray],
        weights_gradients: list[np.ndarray],
        previous_weights_deltas: list[np.ndarray],
        update_params: UpdateParameters
        ) -> list[np.ndarray]:
    """
    Updates weights using gradient descent with L2 regularization
    and momentum.

    Parameters
    ----------
    weights : list of np.ndarray
        List of weights matrices of the neural network.
    weights_gradients : list of np.ndarray
        List containing the negative weights gradient matrix of each layer.
    previous_weights_deltas : list of np.ndarray
        List containing the weights deltas of the previous weights update
        iteration. Needed to apply momentum to the weight update.
    update_params : UpdateParameters
        Contains the parameters needed to perform the weights update.

    Returns
    -------
    weights_deltas : list of np.ndarray
        List containing the delta applied to the weights.
        To be used in future iteration of the training cycle
        as previous_weights_delta.

    Notes
    -----
    The weights update is performed following:
        delta_w = alpha * delta_w_old + eta * (neg_w_grad - lambda * w)
        w_new = w + delta_w
    where `neg_w_grad` is the negative gradient passed in `weights_gradients`.
    """
    weights_deltas = []

    for weight, previous_delta, gradient in zip(
            weights,
            previous_weights_deltas,
            weights_gradients,
            strict=True
            ):
        reg_term = update_params.lambda_ * weight
        weights_delta = (
            update_params.alpha * previous_delta
            + update_params.eta * (gradient - reg_term)
        )
        weight += weights_delta
        weights_deltas.append(weights_delta)

    return weights_deltas
