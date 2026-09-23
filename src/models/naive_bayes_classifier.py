# src/models/naive_bayes_classifier.py

"""
Naive Bayes classifier for phishing email detection.

This module provides a reusable wrapper around scikit-learn's
MultinomialNB model. It is designed to work with the project's
TF-IDF feature matrix and integrate with ModelFactory and the
training pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.naive_bayes import MultinomialNB

logger = logging.getLogger(__name__)


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for binary phishing
    email classification.

    MultinomialNB is suitable for text classification because it
    works with non-negative feature representations such as TF-IDF.

    Attributes:
        model: The underlying scikit-learn MultinomialNB model.
        is_fitted: Indicates whether the model has been trained.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
    ) -> None:
        """
        Initialize the Naive Bayes classifier.

        Args:
            alpha: Additive smoothing parameter. A larger value
                applies stronger smoothing.
            fit_prior: Whether to learn class prior probabilities
                from the training data.

        Raises:
            ValueError: If alpha is negative.
        """
        if alpha < 0:
            raise ValueError("alpha must be greater than or equal to 0.")

        self.model = MultinomialNB(
            alpha=alpha,
            fit_prior=fit_prior,
        )

        self.is_fitted = False

        logger.info(
            "Initialized Naive Bayes classifier "
            "(alpha=%s, fit_prior=%s).",
            alpha,
            fit_prior,
        )

    def fit(self, X: Any, y: Any) -> NaiveBayesClassifier:
        """
        Train the Naive Bayes model.

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
            "Training Naive Bayes classifier on %d samples.",
            len(y),
        )

        self.model.fit(X, y)
        self.is_fitted = True

        logger.info("Naive Bayes classifier training completed.")

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
        Generate class probability estimates.

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

    @property
    def classes_(self) -> Any:
        """
        Return the class labels learned by the underlying model.

        Returns:
            Class labels learned during training.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self._validate_fitted()
        return self.model.classes_

    def _validate_fitted(self) -> None:
        """
        Ensure that the classifier has been trained before inference.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Naive Bayes classifier must be fitted before "
                "making predictions."
            )