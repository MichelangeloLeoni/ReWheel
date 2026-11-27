from ..classes import EvaluationParams
from .loss import (
    compute_categorical_cross_entropy,
    compute_hmse,
    compute_binary_cross_entropy
)
from .classification_metrics import (
    compute_accuracy,
    compute_confusion_matrix
)

def cce(
        params: EvaluationParams
        ) -> float:

    return compute_categorical_cross_entropy(
        output_vectors=params.output_vector,
        target_vector=params.target_vector
    )

def bce(
        params: EvaluationParams
        ) -> float:

    return compute_binary_cross_entropy(
        output_vectors=params.output_vector,
        target_vector=params.target_vector
    )

def hmse(
        params: EvaluationParams
        ) -> float:

    return compute_hmse(
        output_vectors=params.output_vector,
        target_vector=params.target_vector
    )

def acc(
        params: EvaluationParams
        ) -> float:

    return compute_accuracy(
        predicted_classes=params.predicted_classes,
        true_classes=params.true_classes,
        multilabel=params.multilabel
    )

def cm(
        params: EvaluationParams
        ) -> float:

    return compute_confusion_matrix(
        predicted_classes=params.predicted_classes,
        true_classes=params.true_classes,
        n_classes=params.n_classes,
        multilabel=params.multilabel
    )

loss_functions = {
    'HMSE': hmse,
    'BCE': bce,
    'CCE': cce
}

classification_metrics = {
    'ACC': acc,
    'CM': cm
}

metric_compatibility = {
    'HMSE': ('B', 'ML', 'MC', 'R'),
    'BCE': ('B', 'ML'),
    'CCE': ('MC',),
    'ACC': ('B', 'ML', 'MC'),
    'CM': ('B', 'ML', 'MC')
}
