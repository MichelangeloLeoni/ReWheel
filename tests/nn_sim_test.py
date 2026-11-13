"""
Test module for the nn simulation library.
These tests are AI generated to make things easier, thus they need to be double-checked
"""
import pytest
import numpy as np
from nn_simulator import Architecture, NeuralNetwork, UpdateParameters, TrainingParameters
from nn_simulator.engine.propagation import forward_propagation, backward_propagation, update_weights
from nn_simulator.engine.loss import compute_error, compute_loss
from nn_simulator.engine.utils import initialize_weights
from nn_simulator.engine.training.trainer import Trainer
from nn_simulator.engine.classes import StoppingCriteria

def test_save_and_load_model():
    """Test function for saving and loading a neural network model"""
    test_architecture = Architecture(
        layers=((2, "Sigmoid"), (3, "Sigmoid"), (3, "Sigmoid"), (1, "Sigmoid")),
        input_vector_size=2
        )
    test = NeuralNetwork(architecture=test_architecture)
    test.save_model(model_name="test")
    loaded_test = NeuralNetwork.load_model(model_path="models/test")

    assert isinstance(loaded_test, NeuralNetwork)
    assert loaded_test.architecture.layers == test.architecture.layers
    assert loaded_test.architecture.act_funcs == test.architecture.act_funcs
    assert loaded_test.architecture.input_vector_size == test.architecture.input_vector_size

def test_forward_propagation():
    """Test function for forward_propagation"""
    test_architecture = Architecture(
        layers=((2, "Linear"), (1, "Linear")),
        input_vector_size=2
        )
    test = NeuralNetwork(architecture=test_architecture)
    initialize_weights(test_architecture, test.weights)
    for i, _ in enumerate(test.weights):
        test.weights[i].fill(1/2)

    # Single example test
    input_vec = np.array([1, 1])
    output, nets = forward_propagation(
        weights=test.weights,
        act_funcs=test.architecture.act_funcs,
        input_vector=input_vec
    )
    output = output[-1]
    assert len(nets) == 2
    assert output == 2
    assert output.shape == (1,)

    # Batch test
    input_vec = np.ones((2, 2))
    output, nets = forward_propagation(
        weights=test.weights,
        act_funcs=test.architecture.act_funcs,
        input_vector=input_vec
    )
    output = output[-1]
    assert len(nets) == 2
    assert output.shape == (1, 2)
    np.testing.assert_allclose(output, np.array([[2, 2]]))

def test_compute_error():
    """Test function for compute_error"""

    # Single example
    output = np.array([0.2, 0.5, 0.8])
    target = np.array([0.3, 0.4, 1.0])

    error = compute_error(output, target)

    expected_error = np.array([0.1, -0.1, 0.2])
    np.testing.assert_allclose(error, expected_error)
    assert error.shape == (3,)

    # Single example, target is not a NumPy array
    error = compute_error(output, [0.3, 0.4, 1.0])
    np.testing.assert_allclose(error, expected_error)

    # Batch: shape (n_outputs, n_examples)
    output = np.array([
        [0.2, 0.5],
        [0.5, 0.3]
    ])
    target = np.array([
        [0.3, 0.4],
        [0.4, 0.5]
    ])

    error = compute_error(output, target)

    expected_error = np.array([
        [0.1, -0.1],
        [-0.1, 0.2]
    ])
    np.testing.assert_allclose(error, expected_error)
    assert error.shape == (2, 2)

def test_compute_loss():
    """Test function for compute_loss"""

    # 1. Single example, single output
    output = np.array([0.8])
    target = np.array([1.0])

    loss = compute_loss(output, target)
    expected_loss = 0.5 * (0.2 ** 2)

    np.testing.assert_allclose(loss, expected_loss)

    # 2. Batch of examples, single output
    output = np.array([[0.8, 0.6, 0.9]])
    target = np.array([[1.0, 0.5, 0.7]])

    loss = compute_loss(output, target)
    expected_loss = 0.5 * np.mean([
        0.2**2,
        (-0.1)**2,
        (-0.2)**2
    ])

    np.testing.assert_allclose(loss, expected_loss)

    # 3. Single example, multiple outputs
    output = np.array([0.8, 0.6, 0.9])
    target = np.array([1.0, 0.5, 0.7])

    loss = compute_loss(output, target)
    expected_loss = 0.5 * (
        0.2**2 + (-0.1)**2 + (-0.2)**2
    )

    np.testing.assert_allclose(loss, expected_loss)

    # 4. Batch of examples, multiple outputs
    output = np.array([
        [0.8, 0.6, 0.9],
        [0.4, 0.7, 0.2]
    ])
    target = np.array([
        [1.0, 0.5, 0.7],
        [0.5, 0.9, 0.3]
    ])

    loss = compute_loss(output, target)
    expected_loss = 0.5 * np.mean([
        0.2**2 + 0.1**2,
        (-0.1)**2 + 0.2**2,
        (-0.2)**2 + 0.1**2
    ])

    np.testing.assert_allclose(loss, expected_loss)



def test_backward_propagation():
    """Test function for backward_propagation"""

    # 1. Single example, single output
    architecture = Architecture(
        layers=((2, "Linear"), (1, "Linear")),
        input_vector_size=2
    )
    nn = NeuralNetwork(architecture=architecture)
    initialize_weights(architecture, nn.weights)

    for weights in nn.weights:
        weights.fill(0.5)

    input_vec = np.array([1.0, 1.0])
    target = np.array([2.4])

    output, nets = forward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        input_vector=input_vec
    )

    np.testing.assert_allclose(output[-1], [2.0])

    error = compute_error(output, target)

    gradients = backward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        output_vectors=output,
        computed_nets=nets,
        error_vector=error
    )

    expected_gradients = [
        np.full((2, 3), 0.2),
        np.array([[0.6, 0.6, 0.4]])
    ]

    assert len(gradients) == len(nn.weights)

    for gradient, weights, expected in zip(
        gradients, nn.weights, expected_gradients
    ):
        assert gradient.shape == weights.shape
        np.testing.assert_allclose(gradient, expected)

    # Helper: mean of individual-example gradients
    def mean_single_gradients(weights_list, act_funcs, input_batch, target_batch):
        individual_gradients = []

        for i in range(input_batch.shape[1]):
            output, nets = forward_propagation(
                weights=weights_list,
                act_funcs=act_funcs,
                input_vector=input_batch[:, i]
            )

            error = compute_error(output, target_batch[:, i])

            gradients = backward_propagation(
                weights=weights_list,
                act_funcs=act_funcs,
                output_vectors=output,
                computed_nets=nets,
                error_vector=error
            )
            individual_gradients.append(gradients)

        return [
            np.mean(
                [gradients[layer] for gradients in individual_gradients],
                axis=0
            )
            for layer in range(len(weights_list))
        ]

    # 2. Batch of examples, single output
    input_batch = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0]
    ])
    target_batch = np.array([[2.4, 3.0, 4.0]])

    output, nets = forward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        input_vector=input_batch
    )

    error = compute_error(output, target_batch)

    gradients = backward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        output_vectors=output,
        computed_nets=nets,
        error_vector=error
    )

    expected_gradients = mean_single_gradients(
        nn.weights, nn.architecture.act_funcs, input_batch, target_batch
    )

    for gradient, weights, expected in zip(
        gradients, nn.weights, expected_gradients
    ):
        assert gradient.shape == weights.shape
        np.testing.assert_allclose(gradient, expected)

    # 3. Single example, multiple outputs
    architecture = Architecture(
        layers=((2, "Linear"), (3, "Linear")),
        input_vector_size=2
    )
    nn = NeuralNetwork(architecture=architecture)
    initialize_weights(architecture, nn.weights)

    for weights in nn.weights:
        weights.fill(0.5)

    input_vec = np.array([1.0, 1.0])
    target = np.array([2.4, 2.0, 1.5])

    output, nets = forward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        input_vector=input_vec
    )

    np.testing.assert_allclose(
        output[-1], [2.0, 2.0, 2.0]
    )

    error = compute_error(output, target)

    gradients = backward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        output_vectors=output,
        computed_nets=nets,
        error_vector=error
    )

    expected_gradients = [
        np.full((2, 3), -0.05),
        np.array([
            [0.6, 0.6, 0.4],
            [0.0, 0.0, 0.0],
            [-0.75, -0.75, -0.5]
        ])
    ]

    assert len(gradients) == len(nn.weights)

    for gradient, weights, expected in zip(
        gradients, nn.weights, expected_gradients
    ):
        assert gradient.shape == weights.shape
        np.testing.assert_allclose(gradient, expected)

    # 4. Batch of examples, multiple outputs
    input_batch = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0]
    ])
    target_batch = np.array([
        [2.4, 3.0, 4.0],
        [2.0, 1.5, 2.5],
        [1.5, 2.0, 3.0]
    ])

    output, nets = forward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        input_vector=input_batch
    )

    error = compute_error(output, target_batch)

    gradients = backward_propagation(
        weights=nn.weights,
        act_funcs=nn.architecture.act_funcs,
        output_vectors=output,
        computed_nets=nets,
        error_vector=error
    )

    expected_gradients = mean_single_gradients(
        nn.weights, nn.architecture.act_funcs, input_batch, target_batch
    )

    for gradient, weights, expected in zip(
        gradients, nn.weights, expected_gradients
    ):
        assert gradient.shape == weights.shape
        np.testing.assert_allclose(gradient, expected)

def test_update_weights():
    """Test function for update_weights."""

    initial_weights = [
        np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]]),
        np.array([[0.5, -0.5, 1.0]])
    ]

    weights_gradients = [
        np.array([[0.2, -0.4, 0.6],
                  [-0.2, 0.4, -0.6]]),
        np.array([[0.1, -0.3, 0.5]])
    ]

    previous_weights_deltas = [
        np.array([[0.1, 0.2, 0.3],
                  [-0.1, -0.2, -0.3]]),
        np.array([[0.05, -0.05, 0.1]])
    ]

    eta = 0.5
    alpha = 0.2
    lambda_ = 0.1

    # Test 1: Gradient descent only, without momentum or regularization.
    weights = [w.copy() for w in initial_weights]

    params = UpdateParameters(
        eta=eta,
        alpha=0.0,
        lambda_=0.0,
        full_independence=False
    )

    deltas = update_weights(
        weights,
        weights_gradients,
        previous_weights_deltas,
        params
    )

    expected_deltas = [
        eta * gradient for gradient in weights_gradients
    ]
    expected_weights = [
        weight + delta
        for weight, delta in zip(initial_weights, expected_deltas)
    ]

    for delta, expected in zip(deltas, expected_deltas):
        np.testing.assert_allclose(delta, expected)

    for weight, expected in zip(weights, expected_weights):
        np.testing.assert_allclose(weight, expected)

    # Test 2: Momentum enabled, without regularization.
    weights = [w.copy() for w in initial_weights]

    params = UpdateParameters(
        eta=eta,
        alpha=alpha,
        lambda_=0.0,
        full_independence=False
    )

    deltas = update_weights(
        weights,
        weights_gradients,
        previous_weights_deltas,
        params
    )

    expected_deltas = [
        eta * gradient + alpha * previous_delta
        for gradient, previous_delta in zip(
            weights_gradients, previous_weights_deltas
        )
    ]
    expected_weights = [
        weight + delta
        for weight, delta in zip(initial_weights, expected_deltas)
    ]

    for delta, expected in zip(deltas, expected_deltas):
        np.testing.assert_allclose(delta, expected)

    for weight, expected in zip(weights, expected_weights):
        np.testing.assert_allclose(weight, expected)

    # Test 3: Coupled L2 regularization.
    weights = [w.copy() for w in initial_weights]

    params = UpdateParameters(
        eta=eta,
        alpha=alpha,
        lambda_=lambda_,
        full_independence=False
    )

    deltas = update_weights(
        weights,
        weights_gradients,
        previous_weights_deltas,
        params
    )

    expected_deltas = [
        eta * gradient
        + alpha * previous_delta
        - lambda_ * weight
        for weight, gradient, previous_delta in zip(
            initial_weights,
            weights_gradients,
            previous_weights_deltas
        )
    ]
    expected_weights = [
        weight + delta
        for weight, delta in zip(initial_weights, expected_deltas)
    ]

    for delta, expected in zip(deltas, expected_deltas):
        np.testing.assert_allclose(delta, expected)

    for weight, expected in zip(weights, expected_weights):
        np.testing.assert_allclose(weight, expected)

    # Test 4: Decoupled weight decay.
    weights = [w.copy() for w in initial_weights]

    params = UpdateParameters(
        eta=eta,
        alpha=alpha,
        lambda_=lambda_,
        full_independence=True
    )

    deltas = update_weights(
        weights,
        weights_gradients,
        previous_weights_deltas,
        params
    )

    expected_deltas = [
        eta * gradient + alpha * previous_delta
        for gradient, previous_delta in zip(
            weights_gradients, previous_weights_deltas
        )
    ]
    expected_weights = [
        weight + delta - lambda_ * weight
        for weight, delta in zip(initial_weights, expected_deltas)
    ]

    for delta, expected in zip(deltas, expected_deltas):
        np.testing.assert_allclose(delta, expected)

    for weight, expected in zip(weights, expected_weights):
        np.testing.assert_allclose(weight, expected)

def test_predict():
    """Test function for predict."""
    # Test untrained model exception 
    test_architecture = Architecture(
        layers=((2, "Linear"), (1, "Linear")),
        input_vector_size=2
        )
    test = NeuralNetwork(architecture=test_architecture)

    input_vec = np.array([1])
    with pytest.raises(RuntimeError):
        test.predict(input_vec)

    # Test mismatch of input size exception
    initialize_weights(test_architecture, test.weights)
    for i, _ in enumerate(test.weights):
        test.weights[i].fill(1/2)

    with pytest.raises(ValueError):
        test.predict(input_vec)

    # Single example test
    input_vec = np.array([1, 1])
    output = test.predict(input_vec)
    assert output == 2
    assert output.shape == (1,)

    # Batch test
    input_vec = np.ones((2, 2))
    output = test.predict(input_vec)
    assert output.shape == (1, 2)
    np.testing.assert_allclose(output, np.array([[2, 2]]))

def test_train():
    """Test the NeuralNetwork.train() method."""

    architecture = Architecture(
        layers=((2, "Linear"), (1, "Linear")),
        input_vector_size=2
    )

    nn = NeuralNetwork(architecture=architecture)

    # Weights must not be initialized yet.
    assert all(weight is None for weight in nn.weights)

    # Training parameters.
    epochs = 50
    seed = 42

    training_params = TrainingParameters(
        bounds=(-0.5, 0.5),
        batch_size=None,
        seed=seed
    )

    stopping_criteria = StoppingCriteria(
        epochs=epochs,
        min_delta_loss=None,
        min_gradient_norm=None,
        target_loss=None
    )

    update_params = UpdateParameters(
        eta=0.01,
        alpha=0.0,
        lambda_=0.0,
        full_independence=False
    )

    # Training dataset.
    input_vector = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0]
    ])

    target_vector = np.array([
        [2.4, 3.0, 4.0]
    ])

    # Perform training.
    nn.configure(
        update_params=update_params,
        training_params=training_params,
        stopping_crit=stopping_criteria
    )
    nn.train(
        input_vector=input_vector,
        target_vector=target_vector
    )

    # Weights must be initialized.
    assert all(weight is not None for weight in nn.weights)

    for weight in nn.weights:
        assert np.all(np.isfinite(weight))

    # Training parameters must be stored.
    assert nn.update_params is update_params
    assert nn.training_params is training_params

    # Training log.
    log = nn.training_log

    assert log is not None
    assert log.seed == seed
    assert len(log.loss) == epochs
    assert log.stopping_cause == ["Maximum_epochs_reached"]

    # Loss values.
    assert np.all(np.isfinite(log.loss))
    assert log.loss[-1] < log.loss[0]

    # Gradient norms.
    assert len(log.training_state.gradient_norms) == epochs
    assert np.all(np.isfinite(log.training_state.gradient_norms))
    assert all(norm >= 0 for norm in log.training_state.gradient_norms)

    # Loss variation.
    assert len(log.last_delta_loss) == epochs - 1
    assert np.all(np.isfinite(log.last_delta_loss))

    # Training timestamps.
    assert len(log.training_start_stop) == 2
    assert log.training_start_stop[0] <= log.training_start_stop[1]

def test_train_online():
    """Test training with mini-batches."""

    architecture = Architecture(
        layers=((2, "Linear"), (1, "Linear")),
        input_vector_size=2
    )

    nn = NeuralNetwork(architecture=architecture)

    training_params = TrainingParameters(
        bounds=(-0.5, 0.5),
        seed=42,
        batch_size=2
    )

    stopping_crit = StoppingCriteria(
        epochs=10,
        min_delta_loss=None,
        min_gradient_norm=None,
        target_loss=None
    )

    update_params = UpdateParameters(
        eta=0.01,
        alpha=0.0,
        lambda_=0.0,
        full_independence=False
    )

    input_vector = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0]
    ])

    target_vector = np.array([
        [2.4, 3.0, 4.0]
    ])

    nn.configure(
        update_params=update_params,
        training_params=training_params,
        stopping_crit=stopping_crit
    )
    nn.train(
        input_vector=input_vector,
        target_vector=target_vector
    )

    log = nn.training_log

    assert len(log.loss) == 10

    # Every epoch must process all three examples.
    assert all(weight is not None for weight in nn.weights)

    for weight in nn.weights:
        assert np.all(np.isfinite(weight))

    assert np.all(np.isfinite(log.loss))
    assert np.all(np.isfinite(log.training_state.gradient_norms))

def test_train_reproducibility():
    """Test reproducibility with the same random seed."""

    architecture = Architecture(
        layers=((2, "Linear"), (1, "Linear")),
        input_vector_size=2
    )

    training_params = TrainingParameters(
        bounds=(-0.5, 0.5),
        seed=42,
        batch_size=None
    )

    stopping_crit = StoppingCriteria(
        epochs=10,
        min_delta_loss=None,
        min_gradient_norm=None,
        target_loss=None
    )

    update_params = UpdateParameters(
        eta=0.01,
        alpha=0.0,
        lambda_=0.0,
        full_independence=False
    )

    input_vector = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0]
    ])

    target_vector = np.array([
        [2.4, 3.0, 4.0]
    ])

    nn1 = NeuralNetwork(architecture=architecture)
    nn2 = NeuralNetwork(architecture=architecture)

    nn1.configure(
        update_params=update_params,
        training_params=training_params,
        stopping_crit=stopping_crit
    )
    nn1.train(
        input_vector=input_vector,
        target_vector=target_vector
    )

    nn2.configure(
        update_params=update_params,
        training_params=training_params,
        stopping_crit=stopping_crit
    )
    nn2.train(
        input_vector=input_vector,
        target_vector=target_vector
    )

    for w1, w2 in zip(nn1.weights, nn2.weights):
        np.testing.assert_array_equal(w1, w2)

    np.testing.assert_array_equal(
        nn1.training_log.loss,
        nn2.training_log.loss
    )

def test_train_validation_set():
    """Test validation set handling in NeuralNetwork.train()."""

    architecture = Architecture(
        layers=((2, "Linear"), (1, "Linear")),
        input_vector_size=2
    )

    training_params = TrainingParameters(
        bounds=(-0.5, 0.5),
        seed=42,
        batch_size=None
    )

    update_params = UpdateParameters(eta=0.01)

    stopping_crit = StoppingCriteria(
        epochs=5,
        min_delta_valid_loss=(1e-6, 10)
    )

    input_vector = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0]
    ])

    target_vector = np.array([[2.4, 3.0, 4.0]])

    valid_input_vector = np.array([
        [1.5, 2.5],
        [1.0, 2.0]
    ])

    valid_target_vector = np.array([[2.5, 4.0]])

    # 1. Training without early stopping and without validation set.
    nn = NeuralNetwork(architecture=architecture)

    nn.configure(
        update_params=update_params,
        training_params=training_params,
        stopping_crit=StoppingCriteria(epochs=5)
    )

    nn.train(input_vector, target_vector)

    assert all(weight is not None for weight in nn.weights)
    assert nn.training_log.validation_loss is None

    # 2. Training with validation set and early stopping.
    nn = NeuralNetwork(architecture=architecture)

    nn.configure(
        update_params=update_params,
        training_params=training_params,
        stopping_crit=stopping_crit
    )

    nn.train(
        input_vector,
        target_vector,
        valid_input_vector,
        valid_target_vector
    )

    assert all(weight is not None for weight in nn.weights)

    log = nn.training_log

    assert log.validation_loss is not None
    assert len(log.validation_loss) == len(log.loss)
    assert np.all(np.isfinite(log.validation_loss))
    assert log.best_epoch is not None
    assert 1 <= log.best_epoch <= len(log.loss)

    # 3. Only validation input is provided.
    with pytest.raises(ValueError):
        nn.train(
            input_vector,
            target_vector,
            valid_input_vector=valid_input_vector
        )

    # 4. Only validation target is provided.
    with pytest.raises(ValueError):
        nn.train(
            input_vector,
            target_vector,
            valid_target_vector=valid_target_vector
        )

    # 5. Validation vectors provided as Python lists.
    nn = NeuralNetwork(architecture=architecture)

    nn.configure(
        update_params=update_params,
        training_params=training_params,
        stopping_crit=stopping_crit
    )

    nn.train(
        input_vector,
        target_vector,
        valid_input_vector.tolist(),
        valid_target_vector.tolist()
    )

    assert all(weight is not None for weight in nn.weights)
    assert len(nn.training_log.validation_loss) == len(nn.training_log.loss)

def test_training_stopping_criteria(monkeypatch):
    """Test all available training stopping criteria."""

    architecture = Architecture(
        layers=((2, "Linear"), (1, "Linear")),
        input_vector_size=2
    )

    input_vector = np.ones((2, 3))
    target_vector = np.ones((1, 3))

    # Maximum number of epochs
    trainer = Trainer(
        architecture=architecture,
        update_params=UpdateParameters(eta=0.01),
        training_params=TrainingParameters(seed=42),
        stopping_crit=StoppingCriteria(epochs=3),
        weights=[None] * len(architecture.layers)
    )

    def fake_step(*args, **kwargs):
        trainer.training_log.loss.append(1.0)
        trainer.training_state.gradient_norms.append(1.0)

    monkeypatch.setattr(trainer, "_training_step", fake_step)

    log = trainer.training_loop(input_vector, target_vector)

    assert len(log.loss) == 3
    assert log.stopping_cause == ["Maximum_epochs_reached"]

    # Target loss
    trainer = Trainer(
        architecture=architecture,
        update_params=UpdateParameters(eta=0.01),
        training_params=TrainingParameters(seed=42),
        stopping_crit=StoppingCriteria(target_loss=0.5),
        weights=[None] * len(architecture.layers)
    )

    losses = iter([1.0, 0.7, 0.4])

    def fake_step(*args, **kwargs):
        trainer.training_log.loss.append(next(losses))
        trainer.training_state.gradient_norms.append(1.0)

    monkeypatch.setattr(trainer, "_training_step", fake_step)

    log = trainer.training_loop(input_vector, target_vector)

    assert len(log.loss) == 3
    assert log.stopping_cause == ["Target_loss_reached"]

    # Minimum training loss variation
    trainer = Trainer(
        architecture=architecture,
        update_params=UpdateParameters(eta=0.01),
        training_params=TrainingParameters(seed=42),
        stopping_crit=StoppingCriteria(min_delta_loss=(0.01, 2)),
        weights=[None] * len(architecture.layers)
    )

    losses = iter([1.0, 0.995, 0.992])

    def fake_step(*args, **kwargs):
        trainer.training_log.loss.append(next(losses))
        trainer.training_state.gradient_norms.append(1.0)

    monkeypatch.setattr(trainer, "_training_step", fake_step)

    log = trainer.training_loop(input_vector, target_vector)

    assert len(log.loss) == 3
    assert log.stopping_cause == ["Minimum_loss_variation_reached"]

    # Minimum gradient norm
    trainer = Trainer(
        architecture=architecture,
        update_params=UpdateParameters(eta=0.01),
        training_params=TrainingParameters(seed=42),
        stopping_crit=StoppingCriteria(min_gradient_norm=(0.1, 2)),
        weights=[None] * len(architecture.layers)
    )

    gradient_norms = iter([0.2, 0.05, 0.04])

    def fake_step(*args, **kwargs):
        trainer.training_log.loss.append(1.0)
        trainer.training_state.gradient_norms.append(
            next(gradient_norms)
        )

    monkeypatch.setattr(trainer, "_training_step", fake_step)

    log = trainer.training_loop(input_vector, target_vector)

    assert len(log.loss) == 3
    assert log.stopping_cause == ["Minimum_gradient_norm_reached"]

    # Early stopping
    trainer = Trainer(
        architecture=architecture,
        update_params=UpdateParameters(eta=0.01),
        training_params=TrainingParameters(seed=42),
        stopping_crit=StoppingCriteria(
            min_delta_valid_loss=(0.01, 2)
        ),
        weights=[None] * len(architecture.layers)
    )

    losses = iter([1.0, 0.8, 0.6])
    validation_losses = iter([1.0, 0.995, 0.994])

    def fake_step(*args, **kwargs):
        trainer.training_log.loss.append(next(losses))

        validation_loss = next(validation_losses)
        trainer.training_log.validation_loss.append(validation_loss)

        trainer.training_state.gradient_norms.append(1.0)

        # Update early stopping state.
        trainer._update_early_stopping(validation_loss)

        # Simulate a weight update after each epoch.
        for weight in trainer.weights:
            weight += 1.0

    monkeypatch.setattr(trainer, "_training_step", fake_step)

    initial_weights = [w.copy() for w in trainer.weights]

    log = trainer.training_loop(
        input_vector,
        target_vector,
        valid_input_vector=np.ones((2, 2)),
        valid_target_vector=np.ones((1, 2))
    )

    assert len(log.loss) == 3
    assert len(log.validation_loss) == 3

    assert log.stopping_cause == [
        "Minimum_validation_loss_variation_reached (ES)"
    ]

    # First epoch has the lowest validation loss.
    assert log.best_epoch == 1

    # The weights from the best epoch must be restored.
    for weight, initial_weight in zip(trainer.weights, initial_weights):
        np.testing.assert_array_equal(weight, initial_weight)
