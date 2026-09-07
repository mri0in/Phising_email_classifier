# src/pipelines/inference_ppln.py

"""
Inference pipeline for the phishing email classifier.

This module connects the application orchestration layer with the
prediction component.

The pipeline:
    1. Loads the persisted TF-IDF vectorizer and trained model.
    2. Accepts an email message.
    3. Delegates prediction to Predictor.
    4. Returns the prediction through a standardized PipelineResult.

No model training or feature fitting occurs during inference.
"""

import logging
from pathlib import Path
from typing import Any

from src.inference.predictor import Predictor
from src.orchestration.base_ppln import BasePipeline, PipelineResult


class InferencePipeline(BasePipeline):
    """
    Pipeline responsible for running model inference on an email.
    """

    PIPELINE_NAME = "inference"

    def __init__(
        self,
        vectorizer_path: str | Path,
        model_path: str | Path,
        email_text: str,
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            vectorizer_path: Path to the persisted TF-IDF vectorizer.
            model_path: Path to the persisted trained classifier.
            email_text: Email message that should be classified.

        Raises:
            TypeError: If arguments have invalid types.
            ValueError: If email_text is empty.
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

        if not isinstance(email_text, str):
            raise TypeError(
                "email_text must be a string."
            )

        if not email_text.strip():
            raise ValueError(
                "email_text cannot be empty."
            )

        self.vectorizer_path = Path(vectorizer_path)
        self.model_path = Path(model_path)
        self.email_text = email_text

    @property
    def name(self) -> str:
        """
        Return the registered pipeline name.

        Returns:
            str: Pipeline identifier.
        """

        return self.PIPELINE_NAME

    def run(self) -> PipelineResult:
        """
        Execute the inference workflow.

        Returns:
            PipelineResult: Standardized inference result.
        """

        self.logger.info(
            "Starting inference pipeline."
        )

        predictor = Predictor(
            vectorizer_path=self.vectorizer_path,
            model_path=self.model_path,
        )

        prediction_result: dict[str, Any] = predictor.predict(
            self.email_text
        )

        self.logger.info(
            "Inference pipeline completed successfully."
        )

        return PipelineResult(
            pipeline_name=self.name,
            success=True,
            message="Inference completed successfully.",
            metadata=prediction_result,
        )