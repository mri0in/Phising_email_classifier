# src/models/random_forest_classifier.py

"""
Random Forest classifier for phishing email detection.

This module provides a reusable wrapper around scikit-learn's
RandomForestClassifier. It is designed to work with the project's
TF-IDF feature matrix and integrate with ModelFactory and the
training pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class RandomForestEmailClassifier:
    """
    Random Forest classifier for binary phishing email classification.

    The classifier expects already-vectorized feature data, such as
    the TF-IDF matrix produced by the project's preprocessing pipeline.

    Attributes:
        model: The underlying scikit-learn RandomForestClassifier model.
        is_fitted: Indicates whether the model has been trained.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        class_weight: str | None = None,
        n_jobs: int = -1,
    ) -> None:
        """
        Initialize the Random Forest classifier.

        Args:
            n_estimators: Number of decision trees in the forest.
            max_depth: Maximum depth of each decision tree. ``None``
                allows trees to expand until stopping conditions are met.
            min_samples_split: Minimum number of samples required to
                split an internal node.
            min_samples_leaf: Minimum number of samples required at a
                leaf node.
            random_state: Seed used for reproducibility.
            class_weight: Optional class-weighting strategy. For example,
                ``"balanced"`` can be used when classes are imbalanced.
            n_jobs: Number of CPU cores used during training and prediction.
                ``-1`` uses all available cores.

        Raises:
            ValueError: If any numeric parameter has an invalid value.
        """
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0.")

        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 or None.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )

        self.is_fitted = False

        logger.info(
            "Initialized Random Forest classifier "
            "(n_estimators=%s, max_depth=%s, min_samples_split=%s, "
            "min_samples_leaf=%s, random_state=%s, class_weight=%s, "
            "n_jobs=%s).",
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            random_state,
            class_weight,
            n_jobs,
        )

    def fit(self, X: Any, y: Any) -> RandomForestEmailClassifier:
        """
        Train the Random Forest model.

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
            "Training Random Forest classifier on %d samples.",
            len(y),
        )

        self.model.fit(X, y)
        self.is_fitted = True

        logger.info("Random Forest classifier training completed.")

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
            RuntimeError: If the classifier has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Random Forest classifier must be fitted before "
                "making predictions."
            )