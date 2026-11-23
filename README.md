# ReWheel

A lightweight feedforward neural network simulator and toolkit.
**_Because reinventing the wheel is fun!_**

---

ReWheel is a lightweight feedforward neural network simulator implemented from scratch in Python using NumPy.

## Features

- Fully connected feedforward neural networks.
- Configurable network architectures and activation functions.
- Forward and backward propagation.
- Batch and mini-batch gradient descent.
- Momentum and L2 regularization.
- Configurable training, weight-update and stopping parameters.
- Multiple stopping criteria, including validation-based early stopping.
- Training logs and model persistence.

## Installation

### Install from PyPI

```bash
pip install ReWheel
```

### Install from source

Clone the repository:

```bash
git clone <repository-url>
cd ReWheel
```

Install the package and its dependencies:

```bash
pip install .
```

## Quick start

The following example demonstrates how to define, configure, train and use a neural network.

### 1. Define the network architecture

First, import the required classes and define the network architecture.

```python
import numpy as np

from rewheel import (
    Architecture,
    NeuralNetwork,
    TrainingParameters,
    UpdateParameters,
    StoppingCriteria
)

architecture = Architecture(
    input_vector_size=2,
    layers=[
        (4, "Relu"),
        (1, "Sigmoid")
    ]
)
```

The `Architecture` class specifies the number of input features and the number of neurons and activation function for each layer.

### 2. Configure the network

The simulator provides three separate classes to configure the training process.

#### Training parameters

`TrainingParameters` controls network initialization and data batching.

```python
training_parameters = TrainingParameters(
    bounds=(-1, 1),
    seed=42,
    batch_size=32
)
```

Main parameters:

| Parameter | Description |
| --- | --- |
| `bounds` | Interval used for weight initialization (the domain of a uniform distribution).|
| `seed` | Random seed used for reproducibility. If omitted, a seed is generated automatically. |
| `batch_size` | Number of samples used for each weight update. |

#### Weight update parameters

`UpdateParameters` controls the optimization process.

```python
update_parameters = UpdateParameters(
    eta=0.1,
    alpha=0.9,
    lambda_=0.001,
    full_independence=False
)
```

Main parameters:

| Parameter | Description |
| --- | --- |
| `eta` | Learning rate. |
| `alpha` | Momentum coefficient. |
| `lambda_` | L2 regularization coefficient. |
| `full_independence` | Controls the implementation of the weight-update rule. |

#### Stopping criteria

`StoppingCriteria` defines the conditions that determine when the training process should stop.

```python
stopping_criteria = StoppingCriteria(
    epochs=1000,
    target_loss=0.01,
    min_delta_loss=(1e-6, 10),
    min_gradient_norm=(1e-5, 10)
)
```

Available stopping criteria:

| Parameter | Description |
|---|---|
| `epochs` | Maximum number of training epochs. |
| `target_loss` | Stop when the training loss reaches the specified target. |
| `min_delta_loss` | Stop when the variation in training loss remains below the specified threshold for a given number of consecutive epochs. |
| `min_gradient_norm` | Stop when the gradient norm satisfies the specified threshold for a given number of consecutive epochs. |
| `min_delta_valid_loss` | Stop when the validation loss variation remains below the specified threshold for a given number of consecutive epochs. |

The tuple-based criteria use the following format:

```python
(threshold, patience)
```

Where:

- `threshold` is the minimum variation or gradient norm threshold.
- `patience` is the number of consecutive epochs for which the condition must be satisfied before stopping.

At least one stopping criterion must be specified.

#### Initialize and configure the network

Once all parameters have been defined, initialize and configure the network:

```python
nn = NeuralNetwork(
    architecture=architecture
)

nn.configure(
    training_params=training_parameters,
    update_params=update_parameters,
    stopping_crit=stopping_criteria
)
```

### 3. Prepare the data

The simulator expects input and target data as NumPy arrays, with samples arranged in columns.

For a network with two input features and one output neuron:

```python
X = np.array([
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0]
])

y = np.array([
    [0.0, 1.0, 1.0, 0.0]
])
```

Here, each column represents one sample.

The input matrix has shape `(n_features, n_samples)`, while the target matrix has shape `(n_outputs, n_samples)`.

### 4. Train the network

Train the network by passing the input and target datasets to `train()`:

```python
training_log = nn.train(
    input_data=X,
    target_data=y
)
```

The training procedure automatically evaluates the configured stopping criteria and terminates when one of them is satisfied.

The method returns a `TrainingLog` containing information about the training process, including:

- Training loss.
- Initial and final loss.
- Elapsed training time.
- Random seed.
- Stopping cause.
- Training start and stop times.

#### Validation-based early stopping

The simulator also supports validation-based early stopping through the `min_delta_valid_loss` criterion.

To enable it, specify a validation dataset and configure the corresponding stopping criterion:

```python
stopping_criteria = StoppingCriteria(
    epochs=1000,
    min_delta_valid_loss=(1e-5, 10)
)
```

Then provide the validation inputs and targets when training:

```python
training_log = nn.train(
    input_data=X_train,
    target_data=y_train,
    validation_data=X_validation,
    validation_targets=y_validation
)
```

The validation loss is monitored throughout training. If its variation remains below the specified threshold for the configured number of consecutive epochs, early stopping is triggered.

The implementation also keeps track of the best validation loss and restores the corresponding network weights when early stopping occurs.

Validation inputs and targets must always be provided together.

### 5. Make predictions

After training, use `predict()` to obtain the network outputs:

```python
predictions = nn.predict(X)
```

The returned array contains one prediction for each input sample.

### 6. Save and load a model

Trained models can be persisted and loaded for subsequent use.

```python
# Save the trained model
nn.save_model("my_model")

# Load a previously saved model
loaded_nn = NeuralNetwork.load_model("my_model")

# Make predictions using the loaded model
predictions = loaded_nn.predict(X)
```

This allows you to reuse a trained network without having to train it again.

## Documentation

For a complete description of the available classes, methods, parameters and functions, refer to the source code docstrings.