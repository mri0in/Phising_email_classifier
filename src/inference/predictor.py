# src/inference/predictor.py

"""
Prediction component for the phishing email classifier.

This module loads the persisted TF-IDF vectorizer and trained classifier
and provides prediction functionality for individual email messages.

The Predictor is intentionally independent of the API and pipeline
orchestration layers.
"""

from __future__ import annotations

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

    The loaded model must provide:
        - predict()

    For prediction confidence information, the model must provide at least
    one of:
        - predict_proba()
        - decision_function()

    Models with predict_proba() return a probability-based confidence
    value.

    Models without predict_proba(), such as LinearSVC, use their
    decision_function() output as a decision score. A decision score is
    not treated as a probability.
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

        self.logger = logging.getLogger(
            self.__class__.__name__
        )

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

        self.model = joblib.load(
            self.model_path
        )

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

        Every supported classifier must provide predict().

        The model must additionally provide at least one confidence
        mechanism:
            - predict_proba()
            - decision_function()

        Raises:
            TypeError: If the loaded object is not compatible with the
                Predictor.
        """

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Loaded model does not provide a predict method."
            )

        has_probability = hasattr(
            self.model,
            "predict_proba",
        )

        has_decision_function = hasattr(
            self.model,
            "decision_function",
        )

        if not has_probability and not has_decision_function:
            raise TypeError(
                "Loaded model must provide either a predict_proba "
                "or decision_function method."
            )

    @staticmethod
    def _validate_email_text(
        email_text: str,
    ) -> str:
        """
        Validate and normalize an incoming email message.

        Args:
            email_text: Email message to classify.

        Returns:
            Cleaned input text with surrounding whitespace removed.

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

    def _get_probability_confidence(
        self,
        features: Any,
        prediction: Any,
    ) -> float | None:
        """
        Calculate probability-based confidence when supported.

        Args:
            features: TF-IDF feature representation.
            prediction: Predicted class.

        Returns:
            Probability associated with the predicted class, or None when
            the loaded model does not provide predict_proba().
        """

        if not hasattr(self.model, "predict_proba"):
            return None

        probabilities = self.model.predict_proba(
            features
        )[0]

        if not hasattr(self.model, "classes_"):
            raise TypeError(
                "Loaded model provides predict_proba() but does not "
                "provide classes_."
            )

        predicted_class_indices = np.where(
            self.model.classes_ == prediction
        )[0]

        if predicted_class_indices.size == 0:
            raise ValueError(
                "Predicted class was not found in the model classes."
            )

        predicted_class_index = int(
            predicted_class_indices[0]
        )

        return float(
            probabilities[predicted_class_index]
        )

    def _get_decision_score(
        self,
        features: Any,
    ) -> float | None:
        """
        Calculate the model decision score when supported.

        Args:
            features: TF-IDF feature representation.

        Returns:
            Decision score for binary classification, or None when the
            loaded model does not provide decision_function().

        Raises:
            ValueError: If the decision function does not return exactly
                one binary classification score.
        """

        if not hasattr(
            self.model,
            "decision_function",
        ):
            return None

        decision_scores = self.model.decision_function(
            features
        )

        decision_scores = np.asarray(
            decision_scores
        )

        if decision_scores.ndim == 1:
            if decision_scores.size != 1:
                raise ValueError(
                    "Expected exactly one decision score for a single "
                    "input email."
                )

            return float(
                decision_scores[0]
            )

        if decision_scores.ndim == 2:
            if decision_scores.shape[0] != 1:
                raise ValueError(
                    "Expected exactly one decision-score row for a "
                    "single input email."
                )

            if decision_scores.shape[1] != 1:
                raise ValueError(
                    "Expected a single decision score for binary "
                    "classification."
                )

            return float(
                decision_scores[0, 0]
            )

        raise ValueError(
            "Unexpected decision_function output shape: "
            f"{decision_scores.shape}"
        )

    def predict(
        self,
        email_text: str,
    ) -> dict[str, Any]:
        """
        Predict whether an email is safe or phishing.

        Args:
            email_text: Email message to classify.

        Returns:
            Dictionary containing:
                prediction: Human-readable class label.
                confidence: Probability associated with the prediction
                    when predict_proba() is supported, otherwise None.
                decision_score: Decision-function score when
                    predict_proba() is unavailable.
                model_version: Current model identifier.
        """

        validated_text = self._validate_email_text(
            email_text
        )

        # Transform the incoming email using the SAME TF-IDF vectorizer
        # that was fitted exclusively on the training dataset.
        features = self.feature_extractor.transform(
            [validated_text]
        )

        prediction = self.model.predict(
            features
        )[0]

        confidence = self._get_probability_confidence(
            features=features,
            prediction=prediction,
        )

        decision_score = None

        if confidence is None:
            decision_score = self._get_decision_score(
                features
            )

        prediction_label = (
            "phishing"
            if int(prediction) == 1
            else "safe"
        )

        if confidence is not None:
            self.logger.info(
                "Email prediction completed: %s | "
                "confidence=%.4f",
                prediction_label,
                confidence,
            )
        else:
            self.logger.info(
                "Email prediction completed: %s | "
                "decision_score=%.4f",
                prediction_label,
                decision_score,
            )

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "decision_score": decision_score,
            "model_version": "v1",
        }