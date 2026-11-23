"""Module defining the TrainingParameters dataclass."""
from dataclasses import dataclass

@dataclass
class TrainingParameters:
    """
    Store the parameters controlling the neural network training process.

    Parameters
    ----------
    bounds : tuple[float, float], default=(-1, 1)
        Lower and upper bounds of the uniform distribution used
        to initialize the network weights.
    seed : int or None, default=None
        Seed used to initialize the random number generator.
        If None, a seed is generated automatically. The resulting
        seed can be retrieved from the training log and reused
        to reproduce the initialization, provided that the
        architecture and initialization bounds remain unchanged.
    batch_size : int or None, default=None
        Number of training examples processed in each weight update.
        If None, the entire training dataset is processed in a
        single batch. A value of 1 corresponds to online training.
    """
    bounds: tuple[float, float] = (-1, 1)
    seed: int | None = None
    batch_size: int | None = None
