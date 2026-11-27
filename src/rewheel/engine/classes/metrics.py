"""Module defining the Metrics dataclass."""
from dataclasses import dataclass

@dataclass
class Metrics:
    """Stores training and validation metrics."""
    training: dict[str, list] | None = None
    validation: dict[str, list] | None = None

    def record(
            self,
            computed_metrics: dict,
            validation: bool = False
            ) -> None:

        dataset = self.validation if validation else self.training

        for name, value in computed_metrics.items():
            dataset.setdefault(name, []).append(value)
