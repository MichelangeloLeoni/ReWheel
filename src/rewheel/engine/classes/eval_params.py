from dataclasses import dataclass
import numpy as np

@dataclass
class EvaluationParams:
    output_vector: np.ndarray | None = None
    target_vector: np.ndarray | None = None
    predicted_classes: np.ndarray | None = None
    true_classes: np.ndarray | None = None
    multilabel: bool | None = None
    n_classes: int | None = None
