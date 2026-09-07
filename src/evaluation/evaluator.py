# src/evaluation/evaluator.py

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluator:
    """
    Calculate classification metrics for a trained phishing classifier.

    This class is intentionally kept independent of file loading and
    pipeline orchestration. It receives predictions and ground-truth
    labels and returns evaluation metrics.
    """

    def __init__(self) -> None:
        """Initialize the evaluator and configure its logger."""

        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        Validate ground-truth and prediction arrays.

        Args:
            y_true: Ground-truth labels.
            y_pred: Model predictions.

        Raises:
            TypeError: If inputs are not NumPy arrays.
            ValueError: If arrays are empty or have different lengths.
        """

        if not isinstance(y_true, np.ndarray):
            raise TypeError("y_true must be a NumPy array.")

        if not isinstance(y_pred, np.ndarray):
            raise TypeError("y_pred must be a NumPy array.")

        if y_true.size == 0:
            raise ValueError("y_true cannot be empty.")

        if y_pred.size == 0:
            raise ValueError("y_pred cannot be empty.")

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred must contain the same number of samples."
            )

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """
        Calculate classification metrics.

        Args:
            y_true: Ground-truth labels.
            y_pred: Model predictions.

        Returns:
            Dictionary containing accuracy, precision, recall,
            F1 score, and confusion matrix.
        """

        self._validate_inputs(y_true, y_pred)

        self.logger.info(
            "Calculating evaluation metrics for %d samples.",
            len(y_true),
        )

        metrics = {
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ).tolist(),
        }

        self.logger.info(
            "Evaluation completed. Accuracy: %.4f | "
            "Precision: %.4f | Recall: %.4f | F1: %.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        )

        return metrics