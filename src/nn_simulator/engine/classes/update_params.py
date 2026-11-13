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
    alpha : float, default=0
        Momentum coefficient controlling the contribution of the
        previous weight update.
    lambda_ : float, default=0
        L2 regularization coefficient controlling the strength
        of the weight decay.
    full_independence : bool, default=False
        If True, applies regularization independently of the
        gradient and momentum contributions. Otherwise, the
        regularization term is included directly in the weight
        update.

    Notes
    -----
    The weight update depends on the value of
    ``full_independence``.

    If ``full_independence=False``, regularization is included
    directly in the weight update:

        delta_w(t) = -eta * grad_L(w(t))
                     - lambda * w(t)
                     + alpha * delta_w(t-1)

        w(t+1) = w(t) + delta_w(t)

    If ``full_independence=True``, regularization is applied
    independently after the gradient and momentum update:

        delta_w(t) = -eta * grad_L(w(t))
                     + alpha * delta_w(t-1)

        w(t+1) = w(t) + delta_w(t) - lambda * w(t)

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
    alpha: float = 0.
    lambda_: float = 0.
    full_independence: bool = False
