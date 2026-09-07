# src/inference/predictor.py

"""
Prediction component for the phishing email classifier.

This module loads the persisted TF-IDF vectorizer and trained classifier
and provides prediction functionality for individual email messages.

The Predictor is intentionally independent of the API and pipeline
orchestration layers.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features.text_feature_extractor import TextFeatureExtractor


class Predictor:
    """
    Perform phishing email predictions using persisted ML artifacts.

    The Predictor loads:
        1. A fitted TF-IDF vectorizer.
        2. A trained classification model.

    It then converts incoming email text into TF-IDF features and generates
    a classification prediction with an associated confidence score.
    """

    def __init__(
        self,
        vectorizer_path: str | Path,
        model_path: str | Path,
    ) -> None:
        """
        Initialize the predictor.

        Args:
            vectorizer_path: Path to the fitted TF-IDF vectorizer.
            model_path: Path to the trained classification model.

        Raises:
            TypeError: If paths have invalid types.
            ValueError: If paths are empty.
            FileNotFoundError: If artifacts do not exist.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        if not isinstance(vectorizer_path, (str, Path)):
            raise TypeError(
                "vectorizer_path must be a string or Path."
            )

        if not isinstance(model_path, (str, Path)):
            raise TypeError(
                "model_path must be a string or Path."
            )

        self.vectorizer_path = Path(vectorizer_path)
        self.model_path = Path(model_path)

        if not str(self.vectorizer_path).strip():
            raise ValueError(
                "vectorizer_path cannot be empty."
            )

        if not str(self.model_path).strip():
            raise ValueError(
                "model_path cannot be empty."
            )

        self._validate_artifacts()

        self.feature_extractor = TextFeatureExtractor.load(
            self.vectorizer_path
        )

        self.model = joblib.load(self.model_path)

        self._validate_model()

        self.logger.info(
            "Predictor initialized successfully."
        )

    def _validate_artifacts(self) -> None:
        """
        Validate that the required model artifacts exist.

        Raises:
            FileNotFoundError: If an artifact is missing.
            ValueError: If an artifact path is not a file.
        """

        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                "TF-IDF vectorizer not found: "
                f"{self.vectorizer_path}"
            )

        if not self.vectorizer_path.is_file():
            raise ValueError(
                "TF-IDF vectorizer path is not a file: "
                f"{self.vectorizer_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        if not self.model_path.is_file():
            raise ValueError(
                f"Model path is not a file: {self.model_path}"
            )

    def _validate_model(self) -> None:
        """
        Validate that the loaded model exposes the required prediction API.

        The model must provide both `predict` and `predict_proba`.

        Raises:
            TypeError: If the loaded object is not compatible with the
                Predictor.
        """

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Loaded model does not provide a predict method."
            )

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Loaded model does not provide a predict_proba method."
            )

    @staticmethod
    def _validate_email_text(email_text: str) -> str:
        """
        Validate and normalize an incoming email message.

        Args:
            email_text: Email message to classify.

        Returns:
            str: Cleaned input text with surrounding whitespace removed.

        Raises:
            TypeError: If email_text is not a string.
            ValueError: If email_text is empty.
        """

        if not isinstance(email_text, str):
            raise TypeError(
                "email_text must be a string."
            )

        normalized_text = email_text.strip()

        if not normalized_text:
            raise ValueError(
                "email_text cannot be empty."
            )

        return normalized_text

    def predict(self, email_text: str) -> dict[str, Any]:
        """
        Predict whether an email is safe or phishing.

        Args:
            email_text: Email message to classify.

        Returns:
            Dictionary containing:
                prediction: Human-readable class label.
                confidence: Probability associated with the prediction.
                model_version: Current model identifier.
        """

        validated_text = self._validate_email_text(email_text)

        # Transform the incoming email using the SAME TF-IDF vectorizer
        # that was fitted exclusively on the training dataset.
        features = self.feature_extractor.transform(
            [validated_text]
        )

        prediction = self.model.predict(features)[0]

        probabilities = self.model.predict_proba(features)[0]

        # Find the probability associated with the predicted class.
        predicted_class_index = np.where(
            self.model.classes_ == prediction
        )[0][0]

        confidence = float(
            probabilities[predicted_class_index]
        )

        prediction_label = (
            "phishing"
            if int(prediction) == 1
            else "safe"
        )

        self.logger.info(
            "Email prediction completed: %s | confidence=%.4f",
            prediction_label,
            confidence,
        )

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "model_version": "v1",
        }