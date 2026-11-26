"""Module defining the UpdateParameters dataclass."""
from dataclasses import dataclass

@dataclass
class UpdateParameters:
    """
    Store the hyperparameters controlling the network weight updates.

    Parameters
    ----------
    eta : float
        Learning rate controlling the magnitude of gradient-based
        weight updates.
    alpha : float, default=0.0
        Momentum coefficient controlling the contribution of the
        previous weight update.
    lambda_ : float, default=0.0
        L2 regularization coefficient controlling the strength
        of the weight decay.

    Notes
    -----
    The weight update follows gradient descent with momentum and L2
    regularization:

        delta_w(t) = alpha * delta_w(t-1) - eta * (grad_L(w(t)) + lambda * w(t))
        w(t+1) = w(t) + delta_w(t)

    where:

    - ``w(t)`` is the current weight.
    - ``grad_L(w(t))`` is the gradient of the loss function
      with respect to the current weight.
    - ``delta_w(t-1)`` is the previous weight update.
    - ``eta`` is the learning rate.
    - ``alpha`` is the momentum coefficient.
    - ``lambda`` is the regularization coefficient.
    """
    eta: float
    alpha: float = 0.0
    lambda_: float = 0.0
