# src/models/modelfactory.py

"""
Factory for creating supported phishing email classifiers.

The ModelFactory provides a single entry point for the training pipeline
to create different classifier implementations without coupling the
pipeline to individual model classes.
"""

from __future__ import annotations

import logging
from typing import Any, TypeAlias

from src.models.linear_svm_classifier import LinearSVMClassifier
from src.models.logistic_regression_classifier import (
    LogisticRegressionClassifier,
)
from src.models.naive_bayes_classifier import NaiveBayesClassifier
from src.models.random_forest_classifier import RandomForestEmailClassifier


Classifier: TypeAlias = (
    LogisticRegressionClassifier
    | LinearSVMClassifier
    | NaiveBayesClassifier
    | RandomForestEmailClassifier
)


class ModelFactory:
    """
    Factory responsible for creating supported phishing email classifiers.

    The factory isolates model-selection logic from the training pipeline.
    TrainingPipeline only needs to provide a model name and model
    parameters; the factory determines which classifier implementation
    should be instantiated.
    """

    SUPPORTED_MODELS: tuple[str, ...] = (
        "logistic_regression",
        "linear_svm",
        "naive_bayes",
        "random_forest",
    )

    def __init__(self) -> None:
        """Initialize the model factory."""

        self.logger = logging.getLogger(self.__class__.__name__)

    def create_model(
        self,
        model_name: str,
        model_parameters: dict[str, Any] | None = None,
    ) -> Classifier:
        """
        Create a classifier by its registered model name.

        Args:
            model_name: Registered name of the classifier to create.
            model_parameters: Optional model configuration parameters.

        Returns:
            An initialized classifier instance.

        Raises:
            TypeError: If model_name or model_parameters has an invalid type.
            ValueError: If model_name is empty or unsupported.
        """

        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")

        model_name = model_name.strip().lower()

        if not model_name:
            raise ValueError("model_name cannot be empty.")

        if model_parameters is not None and not isinstance(
            model_parameters,
            dict,
        ):
            raise TypeError(
                "model_parameters must be a dictionary or None."
            )

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

        parameters = model_parameters or {}

        model = self._create_model(
            model_name=model_name,
            model_parameters=parameters,
        )

        self.logger.info(
            "Created model '%s' with parameters: %s",
            model_name,
            parameters,
        )

        return model

    def _create_model(
        self,
        model_name: str,
        model_parameters: dict[str, Any],
    ) -> Classifier:
        """
        Instantiate the requested classifier.

        Model-specific parameter handling is kept inside the factory so
        TrainingPipeline does not need to know which constructor accepts
        which parameters.

        Args:
            model_name: Normalized registered model name.
            model_parameters: Parameters supplied by the training pipeline.

        Returns:
            Initialized classifier instance.
        """

        if model_name == "logistic_regression":
            return LogisticRegressionClassifier(
                C=model_parameters.get("C", 1.0),
                max_iter=model_parameters.get("max_iter", 1000),
                random_state=model_parameters.get("random_state", 42),
                class_weight=model_parameters.get(
                    "class_weight",
                    None,
                ),
            )

        if model_name == "linear_svm":
            return LinearSVMClassifier(
                C=model_parameters.get("C", 1.0),
                max_iter=model_parameters.get("max_iter", 1000),
                random_state=model_parameters.get("random_state", 42),
                class_weight=model_parameters.get(
                    "class_weight",
                    None,
                ),
            )

        if model_name == "naive_bayes":
            return NaiveBayesClassifier(
                alpha=model_parameters.get("alpha", 1.0),
                fit_prior=model_parameters.get(
                    "fit_prior",
                    True,
                ),
            )

        if model_name == "random_forest":
            return RandomForestEmailClassifier(
                n_estimators=model_parameters.get(
                    "n_estimators",
                    200,
                ),
                max_depth=model_parameters.get(
                    "max_depth",
                    None,
                ),
                min_samples_split=model_parameters.get(
                    "min_samples_split",
                    2,
                ),
                min_samples_leaf=model_parameters.get(
                    "min_samples_leaf",
                    1,
                ),
                random_state=model_parameters.get(
                    "random_state",
                    42,
                ),
                class_weight=model_parameters.get(
                    "class_weight",
                    None,
                ),
                n_jobs=model_parameters.get(
                    "n_jobs",
                    -1,
                ),
            )

        raise ValueError(
            f"Model '{model_name}' is registered but has no "
            "factory implementation."
        )