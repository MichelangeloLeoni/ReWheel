from dataclasses import dataclass

@dataclass
class Metrics:
    loss: list[float] | None = None
    accuracy: list[float] | None = None
    validation_loss: list[float] | None = None
    validation_accuracy: list[float] | None = None
