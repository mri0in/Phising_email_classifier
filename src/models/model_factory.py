"""Factory for constructing machine-learning classification models."""

import logging
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression


class ModelFactory:
    """Factory responsible for creating supported classifiers."""

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
        """Create a classification model by its registered name."""

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

        if model_name != "logistic_regression":
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

        parameters = model_parameters or {}
        model = LogisticRegression(**parameters)

        self.logger.info(
            "Created model '%s' with parameters: %s",
            model_name,
            parameters,
        )

        return model