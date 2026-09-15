# src/models/model_factory.py

"""
Factory for constructing machine-learning classification models.

This module centralizes model creation so that training pipelines
do not need to know the construction details of individual models.
"""

import logging
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression


class ModelFactory:
    """
    Factory responsible for creating supported classification models.

    The factory centralizes model construction and provides a single
    entry point for selecting a model by name.
    """

    SUPPORTED_MODELS: tuple[str, ...] = (
        "logistic_regression",
    )

    def __init__(self) -> None:
        """Initialize the model factory."""

        self.logger = logging.getLogger(self.__class__.__name__)

    def create_model(
        self,
        model_name: str,
        model_parameters: dict[str, Any] | None = None,
    ) -> ClassifierMixin:
        """
        Create a classification model.

        Args:
            model_name: Name of the model to create.
            model_parameters: Optional model-specific parameters.

        Returns:
            An initialized scikit-learn classifier.

        Raises:
            TypeError: If model_name or model_parameters has an invalid type.
            ValueError: If the model name is unsupported or empty.
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

        parameters = model_parameters or {}

        if model_name == "logistic_regression":
            model = LogisticRegression(**parameters)

        else:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

        self.logger.info(
            "Created model '%s' with parameters: %s",
            model_name,
            parameters,
        )

        return model