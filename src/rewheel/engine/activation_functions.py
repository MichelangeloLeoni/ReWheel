"""Module containing the activation functions and their derivatives."""
import numpy as np

# Linear
def linear(x):
    """Linear activation function."""
    return x

def linear_derivative(x):
    """Derivative of the linear activation function."""
    return np.ones_like(x)

# Sigmoid
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid activation function."""
    s = sigmoid(x)
    return s * (1 - s)

# Hyperbolic tangent
def tanh(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of the hyperbolic tangent activation function."""
    return 1 - np.tanh(x)**2

# ReLU
def relu(x):
    """Rectified Linear Unit activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of the ReLU activation function."""
    return np.where(x > 0, 1, 0)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of the Leaky ReLU activation function."""
    return np.where(x > 0, 1, alpha)

# ELU
def elu(x, alpha=1.0):
    """Exponential Linear Unit activation function."""
    return np.where(x > 0, x, alpha * np.expm1(x))

def elu_derivative(x, alpha=1.0):
    """Derivative of the ELU activation function."""
    return np.where(x > 0, 1, alpha * np.exp(x))

# Softplus
def softplus(x):
    """Softplus activation function."""
    return np.logaddexp(0, x)

def softplus_derivative(x):
    """Derivative of the Softplus activation function."""
    return sigmoid(x)

# Softmax
def softmax(x):
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def softmax_derivative(x):
    """Jacobian matrix of the Softmax activation function."""
    s = softmax(x)
    return np.diag(s) - np.outer(s, s)

# Activation functions dictionary
af_list = {
    "LINEAR": linear,
    "LINEAR_DERIVATIVE": linear_derivative,

    "SIGMOID": sigmoid,
    "SIGMOID_DERIVATIVE": sigmoid_derivative,

    "RELU": relu,
    "RELU_DERIVATIVE": relu_derivative,

    "SOFTMAX": softmax,
    "SOFTMAX_DERIVATIVE": softmax_derivative,
}
