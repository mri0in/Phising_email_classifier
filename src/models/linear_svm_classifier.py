# src/models/linear_svm_classifier.py

"""
Linear SVM classifier for phishing email detection.

This module provides a reusable wrapper around scikit-learn's LinearSVC
model. It is designed to work with the project's TF-IDF feature matrix
and integrate with ModelFactory and the training pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


class LinearSVMClassifier:
    """
    Linear Support Vector Machine (SVM) classifier for binary phishing
    email classification.

    The classifier expects already-vectorized feature data, such as
    the TF-IDF matrix produced by the project's preprocessing pipeline.

    Attributes:
        model: The underlying scikit-learn LinearSVC model.
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
        Initialize the Linear SVM classifier.

        Args:
            C: Regularization parameter. Smaller values apply stronger
                regularization.
            max_iter: Maximum number of iterations for model optimization.
            random_state: Seed used for reproducibility.
            class_weight: Optional class-weighting strategy. For example,
                ``"balanced"`` can be used when classes are imbalanced.

        Raises:
            ValueError: If C or max_iter is not positive.
        """
        if C <= 0:
            raise ValueError("C must be greater than 0.")

        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0.")

        self.model = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
        )

        self.is_fitted = False

        logger.info(
            "Initialized Linear SVM classifier "
            "(C=%s, max_iter=%s, random_state=%s, class_weight=%s).",
            C,
            max_iter,
            random_state,
            class_weight,
        )

    def fit(self, X: Any, y: Any) -> LinearSVMClassifier:
        """
        Train the Linear SVM model.

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

        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training features X and labels y cannot be empty.")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must contain the same number of samples. "
                f"Received X={len(X)}, y={len(y)}."
            )

        logger.info(
            "Training Linear SVM classifier on %d samples.",
            len(y),
        )

        self.model.fit(X, y)
        self.is_fitted = True

        logger.info("Linear SVM classifier training completed.")

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

        if len(X) == 0:
            raise ValueError("Prediction features X cannot be empty.")

        logger.debug(
            "Generating predictions for %d samples.",
            len(X),
        )

        return self.model.predict(X)

    def decision_function(self, X: Any) -> Any:
        """
        Generate decision scores for the input samples.

        LinearSVC does not provide ``predict_proba``. Instead, its
        decision_function returns a confidence score relative to the
        separating hyperplane.

        Args:
            X: Feature matrix containing samples to classify.

        Returns:
            Decision scores for each sample.

        Raises:
            RuntimeError: If the classifier has not been fitted.
            ValueError: If X is None or empty.
        """
        self._validate_fitted()

        if X is None:
            raise ValueError("Prediction features X cannot be None.")

        if len(X) == 0:
            raise ValueError("Prediction features X cannot be empty.")

        logger.debug(
            "Generating decision scores for %d samples.",
            len(X),
        )

        return self.model.decision_function(X)

    def _validate_fitted(self) -> None:
        """
        Ensure that the classifier has been trained before inference.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Linear SVM classifier must be fitted before "
                "making predictions."
            )