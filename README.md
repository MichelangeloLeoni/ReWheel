# nn_simulator

A lightweight neural network simulator implemented in Python using NumPy.

## Features

- Fully connected feedforward neural networks.
- Configurable network architecture and activation functions.
- Forward and backward propagation.
- Mini-batch and batch gradient descent.
- Momentum and L2 regularization.
- Configurable training parameters and stopping criteria.
- Training logs and model persistence.

## Installation

### Install from pip

```bash
pip install nn_simulator
```

### Install from repository

Clone the repository:

```bash
git clone <repository-url>
cd nn_simulator
```

Install the package and its dependencies:

```bash
pip install .
```

## Quick start

```python
from nn_simulator import (
    Architecture,
    NeuralNetwork,
    TrainingParameters,
    UpdateParameters
)

# Define the network architecture
architecture = Architecture(
    input_vector_size=2,
    layers=[
        (4, "Relu"),
        (1, "Sigmoid")
    ]
)

# Configure training parameters
training_parameters = TrainingParameters(
    epochs=1000,
    batch_size=32,
    seed=42
)
# Configure weights update parameters
update_parameters = UpdateParameters(
    eta=0.1,
    alpha=0.9,
    lambda_=0.001
)

# Initialize the neural network
nn = NeuralNetwork(
    architecture=architecture,
)

# Configure the neural network
nn.configure(
    update_params = update_parameters,
    training_params = training_parameters
)

# Train and make predictions
# See the API documentation for the complete method signatures.
```

## Documentation

For further details about the classes, functions and available parameters, see the source code docstrings.