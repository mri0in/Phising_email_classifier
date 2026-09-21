# src/models/logistic_regression_classifier.py
"""
Logistic Regression classifier for phishing email detection.

This module provides a thin, reusable wrapper around scikit-learn's
LogisticRegression model. It is designed to be used by the model
training pipeline and ModelFactory.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier:
    """
    Logistic Regression classifier for binary phishing email classification.

    The classifier expects already-vectorized feature data, such as the
    TF-IDF matrix produced by the project's preprocessing/vectorization
    pipeline.

    Attributes:
        model: The underlying scikit-learn LogisticRegression model.
        is_fitted: Indicates whether the model has been trained.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: str | None = None,
    ) -> None:
        """
        Initialize the Logistic Regression classifier.

        Args:
            C: Inverse of regularization strength. Smaller values apply
                stronger regularization.
            max_iter: Maximum number of iterations for model optimization.
            random_state: Seed used for reproducibility.
            class_weight: Optional class-weighting strategy. For example,
                ``"balanced"`` can be used when classes are imbalanced.

        Raises:
            ValueError: If C is not positive or max_iter is not positive.
        """
        if C <= 0:
            raise ValueError("C must be greater than 0.")

        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0.")

        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
        )

        self.is_fitted = False

        logger.info(
            "Initialized Logistic Regression classifier "
            "(C=%s, max_iter=%s, random_state=%s, class_weight=%s).",
            C,
            max_iter,
            random_state,
            class_weight,
        )

    def fit(self, X: Any, y: Any) -> LogisticRegressionClassifier:
        """
        Train the Logistic Regression model.

        Args:
            X: Training feature matrix, typically a TF-IDF matrix.
            y: Training target labels.

        Returns:
            The fitted classifier instance.

        Raises:
            ValueError: If X or y is empty or incompatible.
        """
        if X is None or y is None:
            raise ValueError("Training features X and labels y cannot be None.")

        if X.shape[0] == 0 or len(y) == 0:
            raise ValueError("Training features X and labels y cannot be empty.")

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must contain the same number of samples. "
                f"Received X={X.shape[0]}, y={len(y)}."
            )

        logger.info(
            "Training Logistic Regression classifier on %d samples.",
            len(y),
        )

        self.model.fit(X, y)
        self.is_fitted = True

        logger.info("Logistic Regression classifier training completed.")

        return self

    def predict(self, X: Any) -> Any:
        """
        Generate class predictions.

        Args:
            X: Feature matrix containing samples to classify.

        Returns:
            Predicted class labels.

        Raises:
            RuntimeError: If the classifier has not been fitted.
            ValueError: If X is None or empty.
        """
        self._validate_fitted()

        if X is None:
            raise ValueError("Prediction features X cannot be None.")

        if X.shape[0] == 0:
            raise ValueError("Prediction features X cannot be empty.")

        logger.debug(
            "Generating predictions for %d samples.",
            X.shape[0],
        )

        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """
        Generate class probabilities.

        Args:
            X: Feature matrix containing samples to classify.

        Returns:
            Probability estimates for each class.

        Raises:
            RuntimeError: If the classifier has not been fitted.
            ValueError: If X is None or empty.
        """
        self._validate_fitted()

        if X is None:
            raise ValueError("Prediction features X cannot be None.")

        if X.shape[0] == 0:
            raise ValueError("Prediction features X cannot be empty.")

        logger.debug(
            "Generating probability predictions for %d samples.",
            X.shape[0],
        )

        return self.model.predict_proba(X)

    def _validate_fitted(self) -> None:
        """
        Ensure that the classifier has been trained before inference.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Logistic Regression classifier must be fitted before "
                "making predictions."
            )