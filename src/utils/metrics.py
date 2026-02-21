"""Metrics computation for model evaluation."""
from typing import Dict, Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class ComputeMetrics:
    """Compute metrics for model evaluation."""

    def __init__(self, task_metrics: list = None):
        """Initialize metrics computer.

        Args:
            task_metrics: List of metric functions
        """
        self.task_metrics = task_metrics or [f1_score, accuracy_score]

    def __call__(self, eval_pred) -> Dict[str, float]:
        """Compute metrics.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dictionary of metric names to values
        """
        predictions, labels = eval_pred

        metrics_dict = {}
        for metric_func in self.task_metrics:
            metric_name = metric_func.__name__
            if metric_name in ["pearsonr", "spearmanr"]:
                score = metric_func(labels, np.squeeze(predictions))
            elif metric_name == "f1_score":
                score = metric_func(
                    np.argmax(predictions, axis=-1), labels, average="macro"
                )
            else:
                score = metric_func(np.argmax(predictions, axis=-1), labels)

            if isinstance(score, tuple):
                metrics_dict[metric_func.__name__] = score[0]
            else:
                metrics_dict[metric_func.__name__] = score

        return metrics_dict


def compute_metrics(
    eval_pred, task_metrics: list = None
) -> Dict[str, float]:
    """Compute metrics for model evaluation.

    Args:
        eval_pred: Tuple of (predictions, labels)
        task_metrics: List of metric functions

    Returns:
        Dictionary of metric names to values
    """
    metrics_computer = ComputeMetrics(task_metrics)
    return metrics_computer(eval_pred)
